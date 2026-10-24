from collections import defaultdict
from datetime import datetime, time, timedelta
from decimal import Decimal
import json

from django.utils import timezone

from .models import BabyLog


ANALYSIS_PERIODS = {
    "24h": {"label": "直近24時間", "days": 1},
    "7d": {"label": "過去7日間", "days": 7},
    "30d": {"label": "過去30日間", "days": 30},
}
MAX_AI_ANALYSIS_BYTES = 32 * 1024
MAX_AI_TIME_SAMPLES_PER_DAY = 2


def _number(value):
    return float(value) if value is not None else None


def _format_timestamp(value):
    return timezone.localtime(value).strftime("%H:%M")


def _empty_day(day):
    return {
        "date": day.isoformat(),
        "record_count": 0,
        "record_counts": {
            log_type: 0 for log_type, _ in BabyLog.LOG_TYPES
        },
        "milk_times": [],
        "milk_amount_ml": None,
        "milk_amount_record_count": 0,
        "milk_amount_complete": False,
        "sleep_durations_minutes": [],
        "sleep_total_minutes": None,
        "sleep_duration_record_count": 0,
        "sleep_duration_complete": False,
        "diaper_times": [],
        "toilet_times": [],
        "temperature_readings": [],
        "other_times": [],
    }


def _period_comparisons(daily_data):
    comparisons = []
    split_at = len(daily_data) // 2
    if split_at < 2 or len(daily_data) - split_at < 2:
        return comparisons

    periods = (
        ("前半", daily_data[:split_at]),
        ("後半", daily_data[split_at:]),
    )
    metrics = (
        ("授乳記録回数", "milk", "record_counts"),
        ("授乳量（ml）", "milk_amount_ml", None),
        ("睡眠記録回数", "sleep", "record_counts"),
        ("睡眠時間（分）", "sleep_total_minutes", None),
        ("おむつ交換記録回数", "diaper", "record_counts"),
        ("排泄記録回数", "toilet", "record_counts"),
        ("体温測定値（℃）", "temperature_mean_c", None),
    )

    for label, metric, nested_key in metrics:
        period_values = []
        for _, days in periods:
            values = []
            for day in days:
                if not day["is_complete_day"]:
                    continue
                if nested_key:
                    value = day[nested_key][metric]
                    if value:
                        values.append(value)
                elif metric == "temperature_mean_c":
                    readings = day["temperature_readings"]
                    if readings:
                        values.append(
                            sum(
                                (Decimal(str(item["temperature_c"])) for item in readings),
                                Decimal(0),
                            )
                            / len(readings)
                        )
                else:
                    value = day[metric]
                    is_complete = (
                        day["milk_amount_complete"]
                        if metric == "milk_amount_ml"
                        else day["sleep_duration_complete"]
                    )
                    if value is not None and is_complete:
                        values.append(value)
            period_values.append(values)

        before, after = period_values
        if len(before) < 2 or len(after) < 2:
            continue

        before_value = sum(before, Decimal(0)) / len(before)
        after_value = sum(after, Decimal(0)) / len(after)
        comparisons.append(
            {
                "metric": label,
                "comparison_basis": (
                    "完了日かつ記録値が揃っている日あたりの平均"
                    if metric in {"milk_amount_ml", "sleep_total_minutes"}
                    else "完了日かつ記録があった日あたりの平均"
                ),
                "before_value": _number(before_value),
                "after_value": _number(after_value),
                "change": _number(after_value - before_value),
                "before_days": len(before),
                "after_days": len(after),
                "before_period": "前半",
                "after_period": "後半",
                "aggregation": "daily_average",
            }
        )
    return comparisons


def _sample_values(values, limit=MAX_AI_TIME_SAMPLES_PER_DAY):
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    indexes = {
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [value for index, value in enumerate(values) if index in indexes]


def build_ai_analysis_payload(analysis_data):
    """Return bounded daily aggregates without duplicating individual logs."""
    record_type_labels = analysis_data["record_type_labels"]
    daily_data = analysis_data["daily_data"]
    daily_summaries = []
    for day in daily_data:
        temperatures = [
            reading["temperature_c"]
            for reading in day["temperature_readings"]
        ]
        temperature_times = [
            reading["time"] for reading in day["temperature_readings"]
        ]
        daily_summary = {
            "date": day["date"],
            "record_count": day["record_count"],
            "is_complete_day": day["is_complete_day"],
            "record_counts": {
                log_type: count
                for log_type, count in day["record_counts"].items()
                if count
            },
        }
        if day["milk_amount_record_count"]:
            daily_summary.update(
                {
                    "milk_amount_ml": day["milk_amount_ml"],
                    "milk_amount_record_count": (
                        day["milk_amount_record_count"]
                    ),
                    "milk_amount_complete": day["milk_amount_complete"],
                }
            )
        if day["milk_times"]:
            daily_summary["milk_times"] = _sample_values(day["milk_times"])
            if len(day["milk_times"]) > MAX_AI_TIME_SAMPLES_PER_DAY:
                daily_summary["milk_times_sampled"] = True
        if day["sleep_duration_record_count"]:
            daily_summary.update(
                {
                    "sleep_total_minutes": day["sleep_total_minutes"],
                    "sleep_duration_record_count": (
                        day["sleep_duration_record_count"]
                    ),
                    "sleep_duration_complete": (
                        day["sleep_duration_complete"]
                    ),
                }
            )
        for log_type, key in (
            ("diaper", "diaper_count"),
            ("toilet", "toilet_count"),
            ("other", "other_count"),
        ):
            if day["record_counts"].get(log_type):
                daily_summary[key] = day["record_counts"][log_type]
        if temperatures:
            daily_summary.update(
                {
                    "temperature_reading_count": len(temperatures),
                    "temperature_min_c": min(temperatures),
                    "temperature_mean_c": day["temperature_mean_c"],
                    "temperature_max_c": max(temperatures),
                    "temperature_values_sampled_c": _sample_values(temperatures),
                    "temperature_times_sampled": _sample_values(
                        temperature_times
                    ),
                }
            )
        daily_summaries.append(daily_summary)

    recording_coverage = {}
    for log_type, label in record_type_labels.items():
        days_with_records = sum(
            day["record_counts"][log_type] > 0 for day in daily_data
        )
        recording_coverage[label] = {
            "days_with_records": days_with_records,
            "days_in_period": len(daily_data),
            "record_count": analysis_data["record_counts"][log_type],
        }

    payload = {
        "period": analysis_data["period"],
        "period_label": analysis_data["period_label"],
        "period_start": analysis_data["period_start"],
        "period_end": analysis_data["period_end"],
        "interval_convention": analysis_data["interval_convention"],
        "record_count": analysis_data["record_count"],
        "record_counts": analysis_data["record_counts"],
        "record_type_labels": analysis_data["record_type_labels"],
        "recording_coverage": recording_coverage,
        "complete_calendar_days": sum(
            day["is_complete_day"] for day in daily_data
        ),
        "partial_calendar_days": [
            day["date"] for day in daily_data if not day["is_complete_day"]
        ],
        "count_semantics": analysis_data["count_semantics"],
        "comparison_method": analysis_data["comparison_method"],
        "missing_day_counts": {
            log_type: len(dates)
            for log_type, dates in analysis_data["missing_days_by_type"].items()
        },
        "measurement_gap_day_counts": {
            metric: len(dates)
            for metric, dates in analysis_data["measurement_gaps"].items()
        },
        "omitted_daily_values_meaning": (
            "数値や項目が日別データにない場合、その日の記録がないか、"
            "対応する値が入力されていません。ゼロ実績を意味しません。"
        ),
        "comparisons": analysis_data["comparisons"],
        "daily_data": daily_summaries,
    }

    payload_bytes = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if payload_bytes > MAX_AI_ANALYSIS_BYTES:
        for day in payload["daily_data"]:
            day["milk_times"] = _sample_values(day["milk_times"], limit=1)
            day["temperature_values_sampled_c"] = _sample_values(
                day["temperature_values_sampled_c"],
                limit=1,
            )
            day["temperature_times_sampled"] = _sample_values(
                day["temperature_times_sampled"],
                limit=1,
            )
        payload_bytes = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    if payload_bytes > MAX_AI_ANALYSIS_BYTES:
        raise ValueError("AI analysis summary exceeds the configured size limit.")

    return payload


def build_baby_log_analysis(user, period="7d", now=None):
    if period not in ANALYSIS_PERIODS:
        raise ValueError(f"Unsupported analysis period: {period}")

    now = now or timezone.now()
    local_now = timezone.localtime(now)
    current_timezone = timezone.get_current_timezone()
    period_info = ANALYSIS_PERIODS[period]
    if period == "24h":
        start = now - timedelta(hours=24)
    else:
        first_day = local_now.date() - timedelta(days=period_info["days"] - 1)
        start = timezone.make_aware(
            datetime.combine(first_day, time.min),
            current_timezone,
        )
    end = now

    logs = list(
        BabyLog.objects.filter(
            user=user,
            timestamp__gte=start,
            timestamp__lt=end,
        )
        .order_by("timestamp", "pk")
        .values(
            "log_type",
            "timestamp",
            "amount",
            "duration",
            "temperature",
        )
    )

    first_day = timezone.localtime(start).date()
    last_day = timezone.localtime(end).date()
    days = {}
    current_day = first_day
    while current_day <= last_day:
        days[current_day] = _empty_day(current_day)
        current_day += timedelta(days=1)

    totals_by_type = {
        log_type: 0 for log_type, _ in BabyLog.LOG_TYPES
    }
    amount_values_by_day = defaultdict(list)
    duration_values_by_day = defaultdict(list)
    all_record_times = []

    for log in logs:
        local_timestamp = timezone.localtime(log["timestamp"])
        day = days[local_timestamp.date()]
        log_type = log["log_type"]
        amount = log["amount"]
        duration = log["duration"]
        temperature = log["temperature"]
        day["record_count"] += 1
        day["record_counts"].setdefault(log_type, 0)
        day["record_counts"][log_type] += 1
        totals_by_type.setdefault(log_type, 0)
        totals_by_type[log_type] += 1

        if log_type == "milk":
            day["milk_times"].append(_format_timestamp(log["timestamp"]))
            if amount is not None:
                amount_values_by_day[local_timestamp.date()].append(amount)
        elif log_type == "sleep":
            if duration is not None:
                duration_values_by_day[local_timestamp.date()].append(
                    duration
                )
                day["sleep_durations_minutes"].append(duration)
        elif log_type == "diaper":
            day["diaper_times"].append(_format_timestamp(log["timestamp"]))
        elif log_type == "toilet":
            day["toilet_times"].append(_format_timestamp(log["timestamp"]))
        elif log_type == "temperature":
            if temperature is not None:
                day["temperature_readings"].append(
                    {
                        "time": _format_timestamp(log["timestamp"]),
                        "temperature_c": _number(temperature),
                    }
                )
        else:
            day["other_times"].append(
                {
                    "time": _format_timestamp(log["timestamp"]),
                    "type": log_type,
                }
            )

        all_record_times.append(
            {
                "date": local_timestamp.date().isoformat(),
                "time": local_timestamp.strftime("%H:%M"),
                "type": log_type,
                "amount_ml": amount if log_type == "milk" else None,
                "duration_minutes": (
                    duration if log_type == "sleep" else None
                ),
                "temperature_c": (
                    _number(temperature)
                    if log_type == "temperature"
                    else None
                ),
            }
        )

    for day_date, day in days.items():
        day_start = timezone.make_aware(
            datetime.combine(day_date, time.min),
            current_timezone,
        )
        next_day_start = day_start + timedelta(days=1)
        day["is_complete_day"] = start <= day_start and end >= next_day_start

        amount_values = amount_values_by_day[day_date]
        milk_count = day["record_counts"]["milk"]
        if amount_values:
            day["milk_amount_ml"] = sum(amount_values)
            day["milk_amount_record_count"] = len(amount_values)
        day["milk_amount_complete"] = (
            milk_count > 0 and len(amount_values) == milk_count
        )

        duration_values = duration_values_by_day[day_date]
        sleep_count = day["record_counts"]["sleep"]
        if duration_values:
            day["sleep_total_minutes"] = sum(duration_values)
            day["sleep_duration_record_count"] = len(duration_values)
        day["sleep_duration_complete"] = (
            sleep_count > 0 and len(duration_values) == sleep_count
        )

        readings = day["temperature_readings"]
        day["temperature_mean_c"] = (
            _number(
                sum(
                    (Decimal(str(item["temperature_c"])) for item in readings),
                    Decimal(0),
                )
                / len(readings)
            )
            if readings
            else None
        )

    daily_data = list(days.values())
    missing_days_by_type = {
        log_type: [
            day["date"]
            for day in daily_data
            if day["record_counts"][log_type] == 0
        ]
        for log_type, _ in BabyLog.LOG_TYPES
    }
    measurement_gaps = {
        "授乳量": [
            day["date"]
            for day in daily_data
            if day["record_counts"]["milk"] > 0
            and not day["milk_amount_complete"]
        ],
        "睡眠時間": [
            day["date"]
            for day in daily_data
            if day["record_counts"]["sleep"] > 0
            and not day["sleep_duration_complete"]
        ],
        "体温値": [
            day["date"]
            for day in daily_data
            if day["record_counts"]["temperature"] > 0
            and not day["temperature_readings"]
        ],
    }

    records_truncated = len(all_record_times) > 120
    if records_truncated:
        sample_indexes = {
            round(index * (len(all_record_times) - 1) / 119)
            for index in range(120)
        }
        individual_records = [
            record
            for index, record in enumerate(all_record_times)
            if index in sample_indexes
        ]
    else:
        individual_records = all_record_times

    return {
        "period": period,
        "period_label": period_info["label"],
        "period_start": timezone.localtime(start).strftime("%Y-%m-%d %H:%M"),
        "period_end": timezone.localtime(end).strftime("%Y-%m-%d %H:%M"),
        "interval_convention": "[開始時刻, 終了時刻)",
        "record_count": len(logs),
        "record_counts": totals_by_type,
        "record_type_labels": dict(BabyLog.LOG_TYPES),
        "count_semantics": (
            "件数は保存された記録行の数です。記録0件は実際に出来事がなかった"
            "ことを意味せず、未記録として扱います。"
        ),
        "comparison_method": (
            "暦日を前半・後半に分け、比較する指標の値が両方の期間で"
            "それぞれ2日以上ある場合だけ比較値を出します。"
        ),
        "daily_data": daily_data,
        "missing_days_by_type": missing_days_by_type,
        "measurement_gaps": measurement_gaps,
        "comparisons": _period_comparisons(daily_data),
        "individual_records": individual_records,
        "individual_records_truncated": records_truncated,
    }
