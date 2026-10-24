# backend/logs/views.py
import calendar
from datetime import datetime, time, timedelta
from decimal import Decimal
import hashlib
import json
import logging
import re

from django.conf import settings
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Q, Sum
from django.db.models.functions import TruncDate
from django.http import HttpResponseBadRequest, HttpResponseForbidden
from django.shortcuts import get_object_or_404, render, redirect
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.dateparse import parse_date
from django.utils import timezone
from .models import BabyLog
from .forms import BabyLogForm, RegistrationForm
from .services import (
    DEFAULT_PREDICTION_REFERENCE_DAYS,
    PREDICTION_REFERENCE_PERIODS,
    predict_next_events,
)
from .ai_service import (
    AI_ADVICE_ERROR_MESSAGE,
    AI_ADVICE_TIMEOUT_MESSAGE,
    generate_baby_care_advice,
)
from .analysis import (
    ANALYSIS_PERIODS,
    build_ai_analysis_payload,
    build_baby_log_analysis,
)
from .ai_service import REPORT_SECTION_KEYS
from .demo_accounts import (
    DEMO_ACCOUNT_GROUP,
    DEMO_ACCOUNT_USERNAMES,
    is_demo_account,
)


SLEEP_DURATION_PATTERN = re.compile(
    r'(?:(?P<hours>\d+(?:\.\d+)?)\s*時間)?\s*'
    r'(?:(?P<minutes>\d+(?:\.\d+)?)\s*分)?'
)
logger = logging.getLogger(__name__)
AI_REPORT_FORMAT_VERSION = 2
MILK_AMOUNT_PATTERN = re.compile(
    r'(?P<amount>\d+(?:\.\d+)?)\s*(?:ml|ミリリットル)',
    re.IGNORECASE,
)


def _sleep_duration_minutes(memo):
    if not memo:
        return Decimal(0)

    total = Decimal(0)
    for match in SLEEP_DURATION_PATTERN.finditer(memo):
        hours = match.group('hours')
        minutes = match.group('minutes')
        if hours is not None or minutes is not None:
            total += Decimal(hours or 0) * 60 + Decimal(minutes or 0)
    return total


def _milk_amount_ml(memo):
    if not memo:
        return Decimal(0)

    match = MILK_AMOUNT_PATTERN.search(memo)
    return Decimal(match.group('amount')) if match else Decimal(0)


def _safe_return_url(request, candidate, fallback):
    if candidate and url_has_allowed_host_and_scheme(
        candidate,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return candidate
    return fallback


def register(request):
    form = RegistrationForm(
        request.POST if request.method == "POST" else None
    )
    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("login")
    return render(request, "register.html", {"form": form})


def demo_login(request):
    if request.method != "POST":
        return HttpResponseForbidden("デモログインはPOSTで実行してください。")
    if not settings.DEMO_ACCOUNTS_ENABLED:
        return HttpResponseForbidden("デモアカウント機能は無効です。")

    account_type = request.POST.get("demo_account")
    username = DEMO_ACCOUNT_USERNAMES.get(account_type)
    if username is None:
        messages.error(request, "デモアカウントを選択できませんでした。")
        return redirect("login")

    user = get_user_model().objects.filter(
        username=username,
        is_active=True,
        groups__name=DEMO_ACCOUNT_GROUP,
    ).first()
    if user is None:
        messages.error(
            request,
            "デモアカウントが初期化されていません。管理者にお問い合わせください。",
        )
        return redirect("login")

    login(request, user, backend="django.contrib.auth.backends.ModelBackend")
    return redirect("log_list")


def _demo_mutation_denied(request):
    if is_demo_account(request.user):
        return HttpResponseForbidden(
            "デモアカウントでは記録の追加・編集・削除はできません。"
        )
    return None


def _handle_log_post(request, fallback_url):
    denied_response = _demo_mutation_denied(request)
    if denied_response is not None:
        return denied_response

    return_url = _safe_return_url(request, request.POST.get('next'), fallback_url)

    if 'quick_log' in request.POST:
        log_type = request.POST.get('quick_log')
        allowed_types = {choice[0] for choice in BabyLog.LOG_TYPES}
        if log_type not in allowed_types:
            messages.error(request, '選択した記録タイプを保存できませんでした。')
            return redirect(return_url)

        quick_log = BabyLog(
            user=request.user,
            log_type=log_type,
            timestamp=timezone.now(),
        )
        if log_type == 'milk':
            quick_log.amount = 190
        quick_log.save()
        messages.success(request, '記録を追加しました。')
        return redirect(return_url)

    form = BabyLogForm(request.POST)
    if form.is_valid():
        log = form.save(commit=False)
        log.user = request.user
        log.save()
        messages.success(request, '記録を保存しました。')
        return redirect(return_url)
    return form


def _get_weekly_analytics_data(user):
    today = timezone.localdate()
    start_date = today - timedelta(days=6)
    current_timezone = timezone.get_current_timezone()
    start_datetime = timezone.make_aware(
        datetime.combine(start_date, time.min),
        current_timezone,
    )
    end_datetime = timezone.make_aware(
        datetime.combine(today + timedelta(days=1), time.min),
        current_timezone,
    )
    daily_totals = (
        BabyLog.objects.filter(
            user=user,
            timestamp__gte=start_datetime,
            timestamp__lt=end_datetime,
        )
        .annotate(date=TruncDate('timestamp', tzinfo=current_timezone))
        .values('date')
        .annotate(
            feeding_amount_ml=Sum('amount', filter=Q(log_type='milk')),
            sleep_duration_minutes=Sum('duration', filter=Q(log_type='sleep')),
            temperature_c=Avg('temperature', filter=Q(log_type='temperature')),
        )
        .order_by('date')
    )
    totals_by_date = {row['date']: row for row in daily_totals}

    daily_data = []
    for offset in range(7):
        day = start_date + timedelta(days=offset)
        totals = totals_by_date.get(day, {})
        temperature = totals.get('temperature_c')
        daily_data.append({
            'date': day.isoformat(),
            'feeding_amount_ml': totals.get('feeding_amount_ml') or 0,
            'sleep_duration_minutes': totals.get('sleep_duration_minutes') or 0,
            'temperature_c': round(float(temperature), 1) if temperature is not None else None,
        })
    return daily_data


@login_required
def log_list(request):
    reference_days_param = request.GET.get('reference_days')
    if reference_days_param is None:
        reference_days = DEFAULT_PREDICTION_REFERENCE_DAYS
    else:
        try:
            reference_days = int(reference_days_param)
        except (TypeError, ValueError):
            return HttpResponseBadRequest('参照期間の指定が正しくありません。')
        if reference_days not in PREDICTION_REFERENCE_PERIODS:
            return HttpResponseBadRequest('参照期間の指定が正しくありません。')

    log_list_url = redirect('log_list').url
    if reference_days != DEFAULT_PREDICTION_REFERENCE_DAYS:
        log_list_url = f'{log_list_url}?reference_days={reference_days}'

    if request.method == 'POST':
        form = _handle_log_post(request, log_list_url)
        if not isinstance(form, BabyLogForm):
            return form
    else:
        form = BabyLogForm()

    logs = BabyLog.objects.filter(user=request.user).order_by('-timestamp')[:8]
    predictions = predict_next_events(request.user, reference_days=reference_days)
    daily_data = _get_weekly_analytics_data(request.user)

    context = {
        'logs': logs,
        'form': form,
        'now': timezone.localtime(),
        'predictions': predictions,
        'prediction_reference_periods': PREDICTION_REFERENCE_PERIODS,
        'daily_data': daily_data,
        'has_feeding_data': any(
            item['feeding_amount_ml'] for item in daily_data
        ),
    }
    return render(request, 'log_list.html', context)


@login_required
def daily_log(request):
    date_param = request.GET.get('date')
    selected_date = parse_date(date_param) if date_param else timezone.localdate()
    if selected_date is None:
        return HttpResponseBadRequest('日付はYYYY-MM-DD形式で指定してください。')

    daily_url = f'{redirect("daily_log").url}?date={selected_date.isoformat()}'
    if request.method == 'POST':
        form = _handle_log_post(request, daily_url)
        if not isinstance(form, BabyLogForm):
            return form
    else:
        form = BabyLogForm()

    current_timezone = timezone.get_current_timezone()
    start = timezone.make_aware(
        datetime.combine(selected_date, time.min),
        current_timezone,
    )
    end = timezone.make_aware(
        datetime.combine(selected_date + timedelta(days=1), time.min),
        current_timezone,
    )
    logs = BabyLog.objects.filter(
        user=request.user,
        timestamp__gte=start,
        timestamp__lt=end,
    ).order_by('timestamp')

    context = {
        'selected_date': selected_date,
        'previous_date': selected_date - timedelta(days=1),
        'next_date': selected_date + timedelta(days=1),
        'logs': logs,
        'form': form,
    }
    return render(request, 'daily_log.html', context)


@login_required
def log_edit(request, pk):
    denied_response = _demo_mutation_denied(request)
    if denied_response is not None:
        return denied_response

    log = get_object_or_404(BabyLog, pk=pk, user=request.user)
    fallback_url = redirect('log_list').url
    return_url = _safe_return_url(
        request,
        request.POST.get('next') or request.GET.get('next'),
        fallback_url,
    )

    if request.method == 'POST':
        form = BabyLogForm(request.POST, instance=log)
        if form.is_valid():
            form.save()
            messages.success(request, '記録を更新しました。')
            return redirect(return_url)
    else:
        form = BabyLogForm(instance=log)

    return render(
        request,
        'log_edit.html',
        {'form': form, 'log': log, 'next_url': return_url},
    )


@login_required
def log_delete(request, pk):
    denied_response = _demo_mutation_denied(request)
    if denied_response is not None:
        return denied_response

    log = get_object_or_404(BabyLog, pk=pk, user=request.user)
    fallback_url = redirect('log_list').url
    return_url = _safe_return_url(
        request,
        request.POST.get('next') or request.GET.get('next'),
        fallback_url,
    )

    if request.method == 'POST':
        log.delete()
        messages.success(request, '記録を削除しました。')
        return redirect(return_url)

    return render(
        request,
        'log_delete_confirm.html',
        {'log': log, 'next_url': return_url},
    )


@login_required
def analytics_dashboard(request):
    daily_data = _get_weekly_analytics_data(request.user)
    return render(
        request,
        'analytics.html',
        {
            'daily_data': daily_data,
            'has_feeding_data': any(
                item['feeding_amount_ml'] for item in daily_data
            ),
            'has_sleep_data': any(
                item['sleep_duration_minutes'] for item in daily_data
            ),
            'has_temperature_data': any(
                item['temperature_c'] is not None for item in daily_data
            ),
        },
    )


@login_required
def ai_advice(request):
    period = (
        request.POST.get("period", "7d")
        if request.method == "POST"
        else request.GET.get("period", "7d")
    )
    if period not in ANALYSIS_PERIODS:
        return HttpResponseBadRequest("分析期間の指定が正しくありません。")

    analysis_data = build_baby_log_analysis(request.user, period)
    session_key = f"ai_advice_report:{request.user.pk}:{period}"
    report_fingerprint = {
        key: value
        for key, value in analysis_data.items()
        if key not in {"period_start", "period_end"}
    }
    report_fingerprint["report_format_version"] = AI_REPORT_FORMAT_VERSION
    report_fingerprint = hashlib.sha256(
        json.dumps(
            report_fingerprint,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    stored_report = request.session.get(session_key, {})
    if not isinstance(stored_report, dict):
        stored_report = {}
    report = (
        stored_report.get("report")
        if stored_report.get("analysis_fingerprint") == report_fingerprint
        else None
    )
    if request.method == 'POST':
        try:
            ai_payload = build_ai_analysis_payload(analysis_data)
        except ValueError as error:
            logger.error(
                "AI analysis payload could not be bounded (%s).",
                type(error).__name__,
            )
            report = AI_ADVICE_ERROR_MESSAGE
        else:
            report = generate_baby_care_advice(ai_payload)
        request.session[session_key] = {
            "analysis_fingerprint": report_fingerprint,
            "report": report,
        }

    report_data = report if isinstance(report, dict) else {}
    reflection_points = "\n".join(
        f"{title}：{section}"
        for title, section in (
            ("今回の確認点", report_data.get("points_to_check", "")),
            ("次回の観察", report_data.get("observation_points", "")),
        )
        if section
    )

    return render(
        request,
        'ai_advice.html',
        {
            'report': report,
            'analysis_data': analysis_data,
            'analysis_periods': ANALYSIS_PERIODS,
            'selected_period': period,
            'report_sections': [
                (
                    "overall_summary",
                    "全体の振り返り",
                    report_data.get("overall_summary", ""),
                ),
                (
                    "item_analysis",
                    "記録状況の評価",
                    report_data.get("item_analysis", ""),
                ),
                (
                    "changes_and_trends",
                    "記録から分かること",
                    report_data.get("changes_and_trends", ""),
                ),
                ("reflection_points", "今後の振り返りポイント", reflection_points),
                (
                    "daily_suggestions",
                    "ひとことアドバイス",
                    report_data.get("daily_suggestions", ""),
                ),
            ],
            'generated': request.method == 'POST',
            'has_recent_logs': bool(analysis_data["record_count"]),
            'advice_timed_out': report == AI_ADVICE_TIMEOUT_MESSAGE,
            'advice_unavailable': (
                isinstance(report, str)
                and report
                in {
                    AI_ADVICE_ERROR_MESSAGE,
                    AI_ADVICE_TIMEOUT_MESSAGE,
                }
            ),
        },
    )


@login_required
def monthly_log(request):
    current_month = timezone.localdate().replace(day=1)
    year_param = request.GET.get('year')
    month_param = request.GET.get('month')

    if year_param is None and month_param is None:
        month_start = current_month
    elif year_param is None or month_param is None:
        return HttpResponseBadRequest('年と月を両方指定してください。')
    else:
        try:
            year = int(year_param)
            month = int(month_param)
            if not 2 <= year <= 9998:
                raise ValueError
            month_start = datetime(year, month, 1).date()
        except (TypeError, ValueError):
            return HttpResponseBadRequest('年または月の指定が正しくありません。')

    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    if month_start.month == 1:
        prev_month = month_start.replace(year=month_start.year - 1, month=12)
    else:
        prev_month = month_start.replace(month=month_start.month - 1)

    current_timezone = timezone.get_current_timezone()
    month_start_datetime = timezone.make_aware(
        datetime.combine(month_start, time.min),
        current_timezone,
    )
    next_month_datetime = timezone.make_aware(
        datetime.combine(next_month, time.min),
        current_timezone,
    )
    logs = list(
        BabyLog.objects.filter(
            user=request.user,
            timestamp__gte=month_start_datetime,
            timestamp__lt=next_month_datetime,
        ).order_by('timestamp')
    )

    recorded_dates = {
        timezone.localtime(log.timestamp).date()
        for log in logs
    }
    calendar_weeks = [
        [
            {
                'date': day,
                'in_month': day.month == month_start.month,
                'has_logs': day in recorded_dates,
            }
            for day in week
        ]
        for week in calendar.Calendar(firstweekday=6).monthdatescalendar(
            month_start.year,
            month_start.month,
        )
    ]

    sleep_total_minutes = sum(
        (
            Decimal(log.duration)
            if log.duration is not None
            else _sleep_duration_minutes(log.memo)
            for log in logs
            if log.log_type == 'sleep'
        ),
        Decimal(0),
    )
    milk_logs = [log for log in logs if log.log_type == 'milk']
    milk_total_ml = sum(
        (
            log.amount
            if log.amount is not None
            else _milk_amount_ml(log.memo)
            for log in milk_logs
        ),
        Decimal(0),
    )
    diaper_count = sum(log.log_type == 'diaper' for log in logs)

    context = {
        'selected_month': month_start,
        'today': timezone.localdate(),
        'prev_month': prev_month,
        'next_month': next_month,
        'calendar_weeks': calendar_weeks,
        'sleep_total_minutes': sleep_total_minutes,
        'sleep_total_hours': sleep_total_minutes / Decimal(60),
        'milk_count': len(milk_logs),
        'milk_total_ml': milk_total_ml,
        'diaper_count': diaper_count,
    }
    return render(request, 'monthly_log.html', context)