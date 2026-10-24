import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal
from requests.exceptions import ReadTimeout, Timeout

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.db import models
from django.core.management import call_command, CommandError
from django.test import Client, TestCase, override_settings
from django.utils import timezone
from django.urls import reverse
from unittest.mock import Mock, patch

from .ai_service import (
    AI_ADVICE_ERROR_MESSAGE,
    generate_baby_care_advice,
)
from .analysis import build_baby_log_analysis
from .analysis import (
    MAX_AI_ANALYSIS_BYTES,
    build_ai_analysis_payload,
)
from .forms import BabyLogForm
from .demo_accounts import (
    DEMO_ACCOUNT_GROUP,
    DEMO_ACCOUNT_USERNAMES,
    is_demo_account,
)
from .models import BabyLog
from .services import (
    DEFAULT_PREDICTION_REFERENCE_DAYS,
    DEFAULT_FEEDING_INTERVAL,
    DEFAULT_WAKE_WINDOW,
    PREDICTION_REFERENCE_PERIODS,
    predict_next_events,
)


SAMPLE_AI_REPORT = {
    "overall_summary": "記録のまとまりを確認しました。",
    "item_analysis": "授乳などの記録を集計しました。",
    "changes_and_trends": "比較できる記録を確認しました。",
    "points_to_check": "未記録の日があります。",
    "observation_points": "同じ項目を継続して記録してください。",
    "daily_suggestions": "無理のない範囲で記録を続けましょう。",
}


class BabyCareAdviceServiceTests(TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_configuration_returns_user_friendly_message(self):
        self.assertEqual(
            generate_baby_care_advice({"feeding_amount_ml": 190}),
            AI_ADVICE_ERROR_MESSAGE,
        )

    @patch("logs.ai_service.requests.post")
    @patch.dict(
        "os.environ",
        {
            "AI_API_URL": "https://ai.example.test/v1/responses",
            "AI_API_KEY": "test-key",
            "AI_MODEL": "",
        },
        clear=True,
    )
    def test_empty_model_uses_default_model(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                SAMPLE_AI_REPORT,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                }
            ]
        }
        mock_post.return_value = mock_response

        result = generate_baby_care_advice({"period": "7d"})

        self.assertEqual(result, SAMPLE_AI_REPORT)
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "gpt-6-luna")

    @patch("logs.ai_service.requests.post")
    @patch.dict(
        "os.environ",
        {
            "AI_API_URL": "https://ai.example.test/v1/chat/completions",
            "AI_API_KEY": "test-key",
            "AI_MODEL": "test-model",
        },
    )
    def test_sends_compatible_chat_request_and_returns_advice(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                SAMPLE_AI_REPORT,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                }
            ]
        }
        mock_post.return_value = mock_response

        advice = generate_baby_care_advice(
            {"期間": "直近7日", "授乳量": 190}
        )

        self.assertEqual(advice, SAMPLE_AI_REPORT)
        mock_post.assert_called_once()
        request = mock_post.call_args
        self.assertEqual(
            request.args[0],
            "https://ai.example.test/v1/chat/completions",
        )
        self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(request.kwargs["json"]["model"], "test-model")
        self.assertEqual(
            request.kwargs["json"]["instructions"][:9],
            "あなたは育児記録の",
        )
        instructions = request.kwargs["json"]["instructions"]
        for heading in (
            "全体の振り返り",
            "記録状況の評価",
            "記録から分かること",
            "今後の振り返りポイント",
            "ひとことアドバイス",
        ):
            self.assertIn(heading, instructions)
        self.assertIn("毎回記録継続を勧める定型文にしません", instructions)
        self.assertEqual(request.kwargs["json"]["max_output_tokens"], 1800)
        self.assertEqual(request.kwargs["timeout"], (5, 45))

    @patch("logs.ai_service.requests.post", side_effect=ReadTimeout("secret"))
    @patch.dict(
        "os.environ",
        {
            "AI_API_URL": "https://ai.example.test/v1/chat/completions",
            "AI_API_KEY": "test-key",
        },
    )
    def test_timeout_returns_user_friendly_message(self, _mock_post):
        self.assertEqual(
            generate_baby_care_advice("記録"),
            "AI分析の応答に時間がかかっています。しばらくしてからもう一度お試しください。",
        )

    @patch("logs.ai_service.requests.post", side_effect=ReadTimeout("secret-value"))
    @patch.dict(
        "os.environ",
        {
            "AI_API_URL": "https://ai.example.test/v1/responses",
            "AI_API_KEY": "secret-value",
        },
    )
    def test_timeout_log_includes_diagnostics_but_not_secrets(self, _mock_post):
        with self.assertLogs("logs.ai_service", level="WARNING") as captured:
            result = generate_baby_care_advice({"period": "7d"})

        self.assertIn("input_bytes=", captured.output[0])
        self.assertIn("connect_timeout=5s", captured.output[0])
        self.assertIn("read_timeout=45s", captured.output[0])
        self.assertNotIn("secret-value", captured.output[0])
        self.assertNotEqual(result, AI_ADVICE_ERROR_MESSAGE)

    @patch("logs.ai_service.requests.post")
    @patch.dict(
        "os.environ",
        {
            "AI_API_URL": "https://ai.example.test/v1/responses",
            "AI_API_KEY": "test-key",
        },
    )
    def test_invalid_or_incomplete_report_returns_friendly_error(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"overall_summary":"only"}'}],
                }
            ]
        }
        mock_post.return_value = mock_response

        self.assertEqual(
            generate_baby_care_advice({"period": "7d"}),
            AI_ADVICE_ERROR_MESSAGE,
        )


class AIAdviceViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="advice-parent",
            password="test-password",
        )
        self.client.force_login(self.user)

    @patch("logs.views.generate_baby_care_advice")
    def test_get_summarizes_recent_logs_without_calling_ai(self, mock_generate):
        now = timezone.now()
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=now - timedelta(hours=2),
            amount=190,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=now - timedelta(hours=4),
            duration=75,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="temperature",
            timestamp=now - timedelta(hours=1),
            temperature="36.7",
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=now - timedelta(hours=25),
            amount=500,
        )
        other_user = get_user_model().objects.create_user(
            username="advice-other",
            password="advice-test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=now - timedelta(hours=1),
            amount=999,
            memo="private note must not be sent",
        )

        response = self.client.get(
            reverse("ai_advice"),
            {"period": "24h"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "ai_advice.html")
        self.assertContains(response, "AI育児アドバイザー")
        self.assertContains(response, "分析レポートを生成する")
        self.assertContains(response, 'class="nav-link active" href="/advice/"')
        self.assertEqual(response.context["analysis_data"]["record_count"], 3)
        self.assertContains(response, "36.7")
        self.assertNotContains(response, "690 ml")
        self.assertNotContains(response, "合計999ml")
        self.assertNotContains(response, "private note must not be sent")
        mock_generate.assert_not_called()

    @patch(
        "logs.views.generate_baby_care_advice",
        return_value=SAMPLE_AI_REPORT,
    )
    def test_post_generates_and_stores_advice_in_session(self, mock_generate):
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timezone.now() - timedelta(minutes=30),
            amount=190,
        )
        other_user = get_user_model().objects.create_user(
            username="advice-other",
            password="advice-test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=timezone.now() - timedelta(minutes=20),
            amount=999,
        )

        response = self.client.post(reverse("ai_advice"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "記録のまとまりを確認しました。")
        mock_generate.assert_called_once()
        summary = mock_generate.call_args.args[0]
        self.assertEqual(summary["period"], "7d")
        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(len(response.context["report_sections"]), 5)
        self.assertEqual(
            [title for _, title, _ in response.context["report_sections"]],
            [
                "全体の振り返り",
                "記録状況の評価",
                "記録から分かること",
                "今後の振り返りポイント",
                "ひとことアドバイス",
            ],
        )
        self.assertIn(
            "今回の確認点：未記録の日があります。",
            response.context["report_sections"][3][2],
        )
        self.assertIn(
            "次回の観察：同じ項目を継続して記録してください。",
            response.context["report_sections"][3][2],
        )
        self.assertNotIn("individual_records", summary)
        self.assertEqual(summary["record_counts"]["milk"], 1)
        self.assertEqual(
            self.client.session[
                f"ai_advice_report:{self.user.pk}:7d"
            ]["report"],
            SAMPLE_AI_REPORT,
        )

    def test_anonymous_user_is_redirected_to_login(self):
        self.client.logout()

        response = self.client.post(reverse("ai_advice"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_default_period_is_seven_days_and_invalid_period_is_rejected(self):
        response = self.client.get(reverse("ai_advice"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["selected_period"], "7d")
        self.assertEqual(
            response.context["analysis_data"]["period_label"],
            "過去7日間",
        )
        self.assertContains(response, 'value="24h"')
        self.assertContains(response, 'value="7d" selected')
        self.assertContains(response, 'value="30d"')

        response = self.client.get(
            reverse("ai_advice"),
            {"period": "all"},
        )
        self.assertEqual(response.status_code, 400)

    @patch(
        "logs.views.generate_baby_care_advice",
        return_value=AI_ADVICE_ERROR_MESSAGE,
    )
    def test_selected_period_and_ai_error_are_shown(self, mock_generate):
        response = self.client.post(
            reverse("ai_advice"),
            {"period": "30d"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["selected_period"], "30d")
        self.assertEqual(response.context["analysis_data"]["period_label"], "過去30日間")
        self.assertContains(response, AI_ADVICE_ERROR_MESSAGE)
        mock_generate.assert_called_once()

    @patch(
        "logs.views.generate_baby_care_advice",
        return_value=SAMPLE_AI_REPORT,
    )
    def test_empty_period_is_sent_as_missing_data_not_fabricated_zeros(
        self, mock_generate
    ):
        response = self.client.post(
            reverse("ai_advice"),
            {"period": "30d"},
        )

        analysis = mock_generate.call_args.args[0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(analysis["record_count"], 0)
        self.assertEqual(len(analysis["daily_data"]), 30)
        self.assertTrue(
            all(
                "milk_amount_ml" not in day
                for day in analysis["daily_data"]
            )
        )
        self.assertIn("ゼロ実績を意味しません", analysis["omitted_daily_values_meaning"])

    @patch(
        "logs.views.generate_baby_care_advice",
        return_value=SAMPLE_AI_REPORT,
    )
    def test_cached_report_is_not_shown_after_source_records_change(
        self, mock_generate
    ):
        self.client.post(reverse("ai_advice"), {"period": "7d"})
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timezone.now(),
            amount=180,
        )

        response = self.client.get(reverse("ai_advice"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "記録のまとまりを確認しました。")
        self.assertEqual(response.context["analysis_data"]["record_count"], 1)
        mock_generate.assert_called_once()

    @patch("logs.views.generate_baby_care_advice")
    def test_cached_report_from_previous_format_is_invalidated(self, mock_generate):
        analysis = build_baby_log_analysis(self.user, "7d")
        old_fingerprint_data = {
            key: value
            for key, value in analysis.items()
            if key not in {"period_start", "period_end"}
        }
        old_fingerprint = hashlib.sha256(
            json.dumps(
                old_fingerprint_data,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        session = self.client.session
        session[f"ai_advice_report:{self.user.pk}:7d"] = {
            "analysis_fingerprint": old_fingerprint,
            "report": SAMPLE_AI_REPORT,
        }
        session.save()

        response = self.client.get(reverse("ai_advice"))

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["report"])
        self.assertNotContains(response, "記録のまとまりを確認しました。")
        mock_generate.assert_not_called()


class BabyLogAnalysisTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="analysis-period-parent",
            password="test-password",
        )
        self.now = timezone.make_aware(datetime(2026, 10, 23, 12, 0))

    def _create_log(self, *, days_ago=0, hour=8, log_type="milk", **values):
        timestamp = timezone.make_aware(
            datetime.combine(
                self.now.date() - timedelta(days=days_ago),
                datetime.min.time(),
            ).replace(hour=hour)
        )
        return BabyLog.objects.create(
            user=values.pop("user", self.user),
            log_type=log_type,
            timestamp=timestamp,
            **values,
        )

    @patch("logs.analysis.timezone.now")
    def test_24_hour_range_has_exact_boundaries_and_only_own_records(self, mock_now):
        mock_now.return_value = self.now
        self._create_log(
            days_ago=1,
            hour=12,
            log_type="milk",
            amount=120,
        )
        self._create_log(
            days_ago=1,
            hour=11,
            log_type="milk",
            amount=999,
        )
        self._create_log(
            log_type="sleep",
            duration=70,
        )
        other_user = get_user_model().objects.create_user(
            username="analysis-period-other",
            password="test-password",
        )
        self._create_log(
            log_type="milk",
            amount=888,
            user=other_user,
        )

        analysis = build_baby_log_analysis(self.user, "24h")

        self.assertEqual(analysis["record_count"], 2)
        self.assertEqual(analysis["record_counts"]["milk"], 1)
        self.assertEqual(analysis["record_counts"]["sleep"], 1)
        self.assertEqual(analysis["period_start"], "2026-10-22 12:00")
        self.assertEqual(analysis["period_end"], "2026-10-23 12:00")
        self.assertEqual(
            analysis["daily_data"][0]["milk_amount_ml"],
            120,
        )
        self.assertNotIn(
            999,
            [
                record["amount_ml"]
                for record in analysis["individual_records"]
            ],
        )

    @patch("logs.analysis.timezone.now")
    def test_7_and_30_day_ranges_include_only_requested_local_calendar_days(
        self, mock_now
    ):
        mock_now.return_value = self.now
        self._create_log(days_ago=6, log_type="milk", amount=100)
        self._create_log(days_ago=7, log_type="milk", amount=900)
        self._create_log(days_ago=29, log_type="sleep", duration=60)
        self._create_log(days_ago=30, log_type="sleep", duration=900)

        seven_day = build_baby_log_analysis(self.user, "7d")
        thirty_day = build_baby_log_analysis(self.user, "30d")

        self.assertEqual(len(seven_day["daily_data"]), 7)
        self.assertEqual(seven_day["record_count"], 1)
        self.assertEqual(seven_day["daily_data"][0]["date"], "2026-10-17")
        self.assertEqual(len(thirty_day["daily_data"]), 30)
        self.assertEqual(thirty_day["record_count"], 3)
        self.assertEqual(thirty_day["period_start"], "2026-09-24 00:00")

    @patch("logs.analysis.timezone.now")
    def test_unrecorded_days_and_missing_measurements_are_not_zero_filled(
        self, mock_now
    ):
        mock_now.return_value = self.now
        self._create_log(days_ago=0, log_type="milk")
        self._create_log(days_ago=0, log_type="sleep")

        analysis = build_baby_log_analysis(self.user, "7d")
        today = analysis["daily_data"][-1]
        previous_day = analysis["daily_data"][-2]

        self.assertEqual(today["record_counts"]["milk"], 1)
        self.assertIsNone(today["milk_amount_ml"])
        self.assertFalse(today["milk_amount_complete"])
        self.assertIsNone(today["sleep_total_minutes"])
        self.assertFalse(today["sleep_duration_complete"])
        self.assertEqual(previous_day["record_count"], 0)
        self.assertIsNone(previous_day["milk_amount_ml"])
        self.assertIsNone(previous_day["sleep_total_minutes"])
        self.assertIn(previous_day["date"], analysis["missing_days_by_type"]["milk"])
        self.assertIn(today["date"], analysis["measurement_gaps"]["授乳量"])

    @patch("logs.analysis.timezone.now")
    def test_comparison_is_only_created_when_each_half_has_enough_recorded_days(
        self, mock_now
    ):
        mock_now.return_value = self.now
        for day in (1, 2, 20, 21):
            self._create_log(
                days_ago=day,
                log_type="milk",
                amount=200 if day > 1 else 100,
            )

        analysis = build_baby_log_analysis(self.user, "30d")

        comparison = next(
            item
            for item in analysis["comparisons"]
            if item["metric"] == "授乳量（ml）"
        )
        self.assertEqual(comparison["before_value"], 200)
        self.assertEqual(comparison["after_value"], 150)
        self.assertEqual(comparison["before_days"], 2)
        self.assertEqual(comparison["after_days"], 2)
        self.assertEqual(comparison["aggregation"], "daily_average")

    @patch("logs.analysis.timezone.now")
    def test_payload_describes_steady_daily_recording_and_excludes_partial_day(
        self, mock_now
    ):
        mock_now.return_value = self.now
        for days_ago in range(7):
            self._create_log(
                days_ago=days_ago,
                log_type="milk",
                amount=120,
            )
            self._create_log(
                days_ago=days_ago,
                hour=14,
                log_type="milk",
                amount=120,
            )

        analysis = build_baby_log_analysis(self.user, "7d")
        payload = build_ai_analysis_payload(analysis)
        milk_comparison = next(
            item
            for item in payload["comparisons"]
            if item["metric"] == "授乳量（ml）"
        )

        self.assertEqual(payload["recording_coverage"]["授乳"]["days_with_records"], 7)
        self.assertEqual(payload["complete_calendar_days"], 6)
        self.assertEqual(payload["partial_calendar_days"], ["2026-10-23"])
        self.assertEqual(milk_comparison["before_value"], 240)
        self.assertEqual(milk_comparison["after_value"], 240)
        self.assertEqual(milk_comparison["before_days"], 3)
        self.assertEqual(milk_comparison["after_days"], 3)

    @patch("logs.analysis.timezone.now")
    def test_payload_comparison_explains_daily_change_not_period_total(
        self, mock_now
    ):
        mock_now.return_value = self.now
        for days_ago, amount in ((5, 100), (4, 200), (2, 200), (1, 400)):
            self._create_log(
                days_ago=days_ago,
                log_type="milk",
                amount=amount,
            )

        payload = build_ai_analysis_payload(
            build_baby_log_analysis(self.user, "7d")
        )
        comparison = next(
            item
            for item in payload["comparisons"]
            if item["metric"] == "授乳量（ml）"
        )

        self.assertEqual(comparison["before_value"], 150)
        self.assertEqual(comparison["after_value"], 300)
        self.assertEqual(comparison["change"], 150)
        self.assertIn("日あたりの平均", comparison["comparison_basis"])

    @patch("logs.analysis.timezone.now")
    def test_payload_marks_missing_categories_without_evaluating_them(
        self, mock_now
    ):
        mock_now.return_value = self.now
        self._create_log(days_ago=1, log_type="milk", amount=120)

        payload = build_ai_analysis_payload(
            build_baby_log_analysis(self.user, "7d")
        )

        self.assertEqual(
            payload["recording_coverage"]["おむつ交換"]["days_with_records"],
            0,
        )
        self.assertEqual(payload["recording_coverage"]["排泄"]["record_count"], 0)
        self.assertEqual(payload["missing_day_counts"]["diaper"], 7)
        self.assertNotIn("diaper_count", payload["daily_data"][0])
        self.assertEqual(payload["recording_coverage"]["授乳"]["days_with_records"], 1)

    @patch("logs.analysis.timezone.now")
    def test_ai_payload_keeps_daily_analysis_and_stays_within_size_limit(
        self, mock_now
    ):
        mock_now.return_value = self.now
        for days_ago in range(30):
            for hour in range(0, 24):
                self._create_log(
                    days_ago=days_ago,
                    hour=hour,
                    log_type="milk",
                    amount=160,
                )
                self._create_log(
                    days_ago=days_ago,
                    hour=hour,
                    log_type="sleep",
                    duration=45,
                )
                self._create_log(
                    days_ago=days_ago,
                    hour=hour,
                    log_type="temperature",
                    temperature="36.7",
                )

        analysis = build_baby_log_analysis(self.user, "30d")
        payload = build_ai_analysis_payload(analysis)
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertLessEqual(len(serialized), MAX_AI_ANALYSIS_BYTES)
        self.assertEqual(len(payload["daily_data"]), 30)
        self.assertEqual(payload["record_count"], (29 * 24 + 12) * 3)
        self.assertNotIn("individual_records", payload)
        self.assertEqual(len(payload["daily_data"][0]["milk_times"]), 2)
        self.assertTrue(payload["daily_data"][0]["milk_times_sampled"])
        self.assertEqual(
            payload["daily_data"][0]["temperature_reading_count"],
            24,
        )
        self.assertIn("milk", payload["missing_day_counts"])


class BabyLogFormTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="form-parent",
            password="form-test-password",
        )

    def test_includes_editable_model_fields(self):
        form = BabyLogForm()

        self.assertEqual(
            set(form.fields),
            {"log_type", "timestamp", "amount", "duration", "temperature", "memo"},
        )

    def test_numeric_and_datetime_widgets_have_expected_attributes(self):
        form = BabyLogForm()

        self.assertEqual(form.fields["amount"].widget.input_type, "number")
        self.assertIsInstance(BabyLog._meta.get_field("amount"), models.IntegerField)
        self.assertEqual(form.fields["amount"].widget.attrs["placeholder"], "ml")
        self.assertEqual(form.fields["duration"].widget.input_type, "number")
        self.assertEqual(form.fields["duration"].widget.attrs["placeholder"], "分")
        self.assertEqual(form.fields["temperature"].widget.input_type, "number")
        self.assertEqual(form.fields["temperature"].widget.attrs["step"], "0.1")
        self.assertEqual(form.fields["temperature"].widget.attrs["placeholder"], "℃")
        self.assertEqual(form.fields["timestamp"].widget.input_type, "datetime-local")

    def test_timestamp_initial_is_current_time(self):
        before = timezone.now()
        form = BabyLogForm()
        after = timezone.now()

        self.assertLessEqual(before, form.initial["timestamp"])
        self.assertLessEqual(form.initial["timestamp"], after)

    def test_timestamp_from_existing_log_is_preserved_for_editing(self):
        timestamp = timezone.now() - timedelta(hours=3)
        log = BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timestamp,
        )

        form = BabyLogForm(instance=log)

        self.assertEqual(
            form.fields["timestamp"].widget.format_value(
                form["timestamp"].value()
            ),
            timezone.localtime(timestamp).strftime("%Y-%m-%dT%H:%M"),
        )


class PredictNextEventsTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="prediction-parent",
            password="prediction-test-password",
        )

    def test_uses_average_feeding_interval_and_sleep_duration(self):
        now = timezone.now()
        first_feeding = now - timedelta(hours=6)
        last_feeding = now - timedelta(hours=3)
        sleep_started = now - timedelta(hours=2)
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=first_feeding,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="feeding",
            timestamp=last_feeding,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=sleep_started,
            duration=60,
        )

        predictions = predict_next_events(user=self.user)

        self.assertEqual(
            predictions["next_feeding"],
            last_feeding + timedelta(hours=3),
        )
        self.assertEqual(
            predictions["next_sleep"],
            sleep_started + timedelta(minutes=60) + DEFAULT_WAKE_WINDOW,
        )

    def test_uses_average_awake_interval_between_sleep_logs(self):
        now = timezone.now()
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=now - timedelta(hours=9),
            duration=60,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=now - timedelta(hours=6),
            duration=90,
        )

        predictions = predict_next_events(user=self.user)

        last_wake_time = now - timedelta(hours=4, minutes=30)
        self.assertEqual(
            predictions["next_sleep"],
            last_wake_time + timedelta(hours=2),
        )

    def test_uses_defaults_when_there_are_no_logs(self):
        before = timezone.now()

        predictions = predict_next_events(user=self.user)

        after = timezone.now()
        self.assertLessEqual(before + DEFAULT_FEEDING_INTERVAL, predictions["next_feeding"])
        self.assertLessEqual(predictions["next_feeding"], after + DEFAULT_FEEDING_INTERVAL)
        self.assertLessEqual(before + DEFAULT_WAKE_WINDOW, predictions["next_sleep"])
        self.assertLessEqual(predictions["next_sleep"], after + DEFAULT_WAKE_WINDOW)
        self.assertFalse(predictions["feeding_data_sufficient"])
        self.assertFalse(predictions["sleep_data_sufficient"])
        self.assertEqual(predictions["feeding_method"], "standard_interval")
        self.assertEqual(predictions["sleep_method"], "standard_wake_window")
        self.assertEqual(predictions["feeding_log_count"], 0)
        self.assertEqual(predictions["sleep_log_count"], 0)
        self.assertEqual(
            predictions["reference_days"],
            DEFAULT_PREDICTION_REFERENCE_DAYS,
        )

    def test_accepts_supported_reference_periods_and_counts_records(self):
        now = timezone.now()
        timestamps = [
            now - timedelta(days=10, hours=1),
            now - timedelta(days=9),
            now - timedelta(hours=1),
        ]
        for timestamp in timestamps:
            BabyLog.objects.create(
                user=self.user,
                log_type="milk",
                timestamp=timestamp,
            )

        seven_day_predictions = predict_next_events(
            user=self.user,
            reference_days=7,
        )
        fourteen_day_predictions = predict_next_events(
            user=self.user,
            reference_days=14,
        )

        self.assertEqual(PREDICTION_REFERENCE_PERIODS, (7, 14, 30, 35, 60))
        self.assertEqual(seven_day_predictions["feeding_log_count"], 1)
        self.assertEqual(seven_day_predictions["feeding_interval_count"], 0)
        self.assertEqual(
            seven_day_predictions["feeding_method"],
            "standard_interval",
        )
        self.assertEqual(fourteen_day_predictions["feeding_log_count"], 3)
        self.assertEqual(fourteen_day_predictions["feeding_interval_count"], 2)
        self.assertTrue(fourteen_day_predictions["feeding_data_sufficient"])
        self.assertEqual(
            fourteen_day_predictions["feeding_method"],
            "observed_average",
        )

    def test_rejects_unsupported_reference_periods(self):
        for reference_days in (0, 8, 90, True, "14"):
            with self.subTest(reference_days=reference_days):
                with self.assertRaises(ValueError):
                    predict_next_events(
                        user=self.user,
                        reference_days=reference_days,
                    )

    def test_sleep_with_no_observed_awake_interval_uses_standard_window(self):
        now = timezone.now()
        sleep_started = now - timedelta(hours=2)
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=sleep_started,
            duration=30,
        )

        predictions = predict_next_events(user=self.user)

        self.assertEqual(predictions["sleep_log_count"], 1)
        self.assertEqual(predictions["sleep_interval_count"], 0)
        self.assertFalse(predictions["sleep_data_sufficient"])
        self.assertEqual(
            predictions["sleep_method"],
            "standard_wake_window",
        )
        self.assertEqual(
            predictions["next_sleep"],
            sleep_started
            + timedelta(minutes=30)
            + DEFAULT_WAKE_WINDOW,
        )

    def test_feeding_logs_older_than_seven_days_are_ignored(self):
        now = timezone.now()
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=now - timedelta(days=8),
        )
        recent_feeding = now - timedelta(minutes=30)
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=recent_feeding,
        )

        predictions = predict_next_events(user=self.user)

        self.assertAlmostEqual(
            (predictions["next_feeding"] - recent_feeding).total_seconds(),
            DEFAULT_FEEDING_INTERVAL.total_seconds(),
            delta=1,
        )

    def test_predictions_ignore_other_users_logs(self):
        other_user = get_user_model().objects.create_user(
            username="other-prediction-parent",
            password="prediction-test-password",
        )
        now = timezone.now()
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=now - timedelta(hours=6),
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=now - timedelta(hours=3),
        )

        predictions = predict_next_events(user=self.user)

        self.assertFalse(predictions["feeding_data_sufficient"])
        self.assertFalse(predictions["sleep_data_sufficient"])
        self.assertEqual(predictions["feeding_log_count"], 0)
        self.assertEqual(predictions["sleep_log_count"], 0)


class AuthenticationFlowTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="parent",
            password="test-password",
        )

    def test_anonymous_user_is_redirected_to_login(self):
        response = self.client.get("/")

        self.assertRedirects(response, "/accounts/login/?next=/")

    def test_successful_login_redirects_to_top(self):
        response = self.client.post(
            "/accounts/login/",
            {"username": "parent", "password": "test-password"},
        )

        self.assertRedirects(response, "/")

    def test_logged_in_user_can_open_top(self):
        self.client.force_login(self.user)

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.context)
        self.assertIn("next_feeding", response.context["predictions"])
        self.assertIn("next_sleep", response.context["predictions"])
        self.assertContains(response, "直近7日間に授乳量の記録はありません。")
        self.assertNotContains(response, 'id="top-feeding-chart"')
        self.assertFalse(response.context["has_feeding_data"])
        self.assertIn("temperature_c", response.context["daily_data"][0])

    def test_top_accepts_reference_period_and_displays_data_status(self):
        self.client.force_login(self.user)

        response = self.client.get("/", {"reference_days": "14"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["predictions"]["reference_days"], 14)
        self.assertEqual(
            response.context["prediction_reference_periods"],
            PREDICTION_REFERENCE_PERIODS,
        )
        self.assertContains(response, 'option value="14" selected')
        self.assertContains(response, 'name="next" value="/?reference_days=14"')
        self.assertContains(response, "14日間の記録 0件・間隔 0件")
        self.assertContains(response, "標準の3時間間隔を使用しています")
        self.assertContains(response, "標準の2時間30分を使用しています")

    def test_top_rejects_unsupported_reference_period(self):
        self.client.force_login(self.user)

        response = self.client.get("/", {"reference_days": "8"})

        self.assertEqual(response.status_code, 400)

    def test_top_lists_only_the_logged_in_users_records(self):
        own_log = BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timezone.now(),
        )
        other_user = get_user_model().objects.create_user(
            username="other-home-parent",
            password="home-test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="sleep",
            timestamp=timezone.now(),
        )
        self.client.force_login(self.user)

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [log.pk for log in response.context["logs"]],
            [own_log.pk],
        )

    def test_logout_redirects_to_login(self):
        self.client.force_login(self.user)

        response = self.client.post("/accounts/logout/")

        self.assertRedirects(response, "/accounts/login/")

    def test_quick_log_saves_current_time_record(self):
        self.client.force_login(self.user)
        before = timezone.now()

        response = self.client.post("/", {"quick_log": "milk"})

        after = timezone.now()
        self.assertRedirects(response, "/")
        log = BabyLog.objects.get(log_type="milk")
        self.assertLessEqual(before, log.timestamp)
        self.assertLessEqual(log.timestamp, after)
        self.assertEqual(log.amount, 190)
        self.assertEqual(log.user, self.user)

    def test_quick_log_rejects_unknown_type(self):
        self.client.force_login(self.user)

        response = self.client.post("/", {"quick_log": "unknown"})

        self.assertRedirects(response, "/")
        self.assertFalse(BabyLog.objects.exists())


class RegistrationFlowTests(TestCase):
    def test_login_page_links_to_registration(self):
        response = self.client.get(reverse("login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("register"))

    def test_user_can_register_and_password_is_hashed(self):
        password = "safely-unique-password-923"
        response = self.client.post(
            reverse("register"),
            {
                "username": "new-parent",
                "password1": password,
                "password2": password,
            },
        )

        self.assertRedirects(response, reverse("login"))
        user = get_user_model().objects.get(username="new-parent")
        self.assertTrue(user.check_password(password))
        self.assertNotEqual(user.password, password)

    def test_mismatched_passwords_do_not_create_a_user(self):
        response = self.client.post(
            reverse("register"),
            {
                "username": "new-parent",
                "password1": "safely-unique-password-923",
                "password2": "another-password-456",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(
            get_user_model().objects.filter(username="new-parent").exists()
        )


class DailyLogTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="daily-parent",
            password="test-password",
        )
        self.client.force_login(self.user)

    def test_shows_only_selected_day_logs_in_time_order(self):
        selected_day = datetime(2026, 10, 15)
        morning = timezone.make_aware(datetime(2026, 10, 15, 8, 30))
        evening = timezone.make_aware(datetime(2026, 10, 15, 21, 0))
        previous_day = timezone.make_aware(datetime(2026, 10, 14, 23, 59))
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=evening,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=morning,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="toilet",
            timestamp=previous_day,
        )
        other_user = get_user_model().objects.create_user(
            username="other-daily-parent",
            password="daily-test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="diaper",
            timestamp=morning,
        )

        response = self.client.get(
            reverse("daily_log"),
            {"date": selected_day.date().isoformat()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            list(response.context["logs"].values_list("log_type", flat=True)),
            ["sleep", "milk"],
        )

    def test_invalid_date_returns_bad_request(self):
        response = self.client.get(reverse("daily_log"), {"date": "not-a-date"})

        self.assertEqual(response.status_code, 400)

    def test_anonymous_user_is_redirected_to_login(self):
        self.client.logout()

        response = self.client.get(reverse("daily_log"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_daily_screen_links_to_selected_month(self):
        response = self.client.get(
            reverse("daily_log"),
            {"date": "2026-10-15"},
        )

        self.assertContains(
            response,
            f'href="{reverse("monthly_log")}?year=2026&amp;month=10"',
        )


class MonthlyLogTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="monthly-parent",
            password="test-password",
        )
        self.client.force_login(self.user)

    def test_monthly_calendar_and_totals(self):
        entries = [
            ("sleep", datetime(2026, 10, 2, 1, 0), "1時間30分"),
            ("sleep", datetime(2026, 10, 2, 22, 0), "45分"),
            ("milk", datetime(2026, 10, 15, 8, 0), "120ml"),
            ("milk", datetime(2026, 10, 15, 12, 0), "90ml"),
            ("diaper", datetime(2026, 10, 15, 13, 0), ""),
            ("toilet", datetime(2026, 10, 15, 14, 0), ""),
            ("milk", datetime(2026, 11, 1, 0, 0), "200ml"),
        ]
        for log_type, timestamp, memo in entries:
            BabyLog.objects.create(
                user=self.user,
                log_type=log_type,
                timestamp=timezone.make_aware(timestamp),
                memo=memo,
            )
        other_user = get_user_model().objects.create_user(
            username="other-monthly-parent",
            password="monthly-test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=timezone.make_aware(datetime(2026, 10, 3, 8, 0)),
            amount=999,
        )

        response = self.client.get(
            reverse("monthly_log"),
            {"year": "2026", "month": "10"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["sleep_total_minutes"], 135)
        self.assertEqual(response.context["milk_count"], 2)
        self.assertEqual(response.context["milk_total_ml"], 210)
        self.assertEqual(response.context["diaper_count"], 1)
        self.assertEqual(response.context["prev_month"], datetime(2026, 9, 1).date())
        self.assertEqual(response.context["next_month"], datetime(2026, 11, 1).date())

        calendar_days = {
            day["date"]: day
            for week in response.context["calendar_weeks"]
            for day in week
        }
        self.assertTrue(calendar_days[datetime(2026, 10, 2).date()]["has_logs"])
        self.assertTrue(calendar_days[datetime(2026, 10, 15).date()]["has_logs"])
        self.assertFalse(calendar_days[datetime(2026, 10, 3).date()]["has_logs"])

    def test_omitted_month_defaults_to_current_month(self):
        response = self.client.get(reverse("monthly_log"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["selected_month"],
            timezone.localdate().replace(day=1),
        )

    def test_invalid_month_returns_bad_request(self):
        response = self.client.get(
            reverse("monthly_log"),
            {"year": "2026", "month": "13"},
        )

        self.assertEqual(response.status_code, 400)

    def test_anonymous_user_is_redirected_to_login(self):
        self.client.logout()

        response = self.client.get(reverse("monthly_log"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])


class LogEditDeleteTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="edit-parent",
            password="test-password",
        )
        self.client.force_login(self.user)
        self.log = BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timezone.now(),
            memo="120ml",
        )

    def test_edit_updates_log_and_returns_to_daily_screen(self):
        daily_url = f'{reverse("daily_log")}?date=2026-10-20'

        response = self.client.post(
            reverse("log_edit", args=[self.log.pk]),
            {
                "log_type": "sleep",
                "timestamp": "2026-10-20 10:30:00",
                "memo": "1時間",
                "next": daily_url,
            },
        )

        self.assertRedirects(response, daily_url)
        self.log.refresh_from_db()
        self.assertEqual(self.log.log_type, "sleep")
        self.assertEqual(self.log.memo, "1時間")

    def test_edit_rejects_external_return_url(self):
        response = self.client.get(
            reverse("log_edit", args=[self.log.pk]),
            {"next": "https://example.com/"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["next_url"], reverse("log_list"))

    def test_delete_get_only_shows_confirmation(self):
        response = self.client.get(
            reverse("log_delete", args=[self.log.pk]),
            {"next": reverse("log_list")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(BabyLog.objects.filter(pk=self.log.pk).exists())

    def test_delete_post_removes_log_and_returns_to_top(self):
        response = self.client.post(
            reverse("log_delete", args=[self.log.pk]),
            {"next": reverse("log_list")},
        )

        self.assertRedirects(response, reverse("log_list"))
        self.assertFalse(BabyLog.objects.filter(pk=self.log.pk).exists())

    def test_anonymous_user_cannot_edit_or_delete(self):
        self.client.logout()

        edit_response = self.client.get(reverse("log_edit", args=[self.log.pk]))
        delete_response = self.client.post(reverse("log_delete", args=[self.log.pk]))

        self.assertEqual(edit_response.status_code, 302)
        self.assertEqual(delete_response.status_code, 302)
        self.assertTrue(BabyLog.objects.filter(pk=self.log.pk).exists())

    def test_edit_saves_structured_measurements(self):
        response = self.client.post(
            reverse("log_edit", args=[self.log.pk]),
            {
                "log_type": "temperature",
                "timestamp": "2026-10-20T10:30",
                "temperature": "36.7",
                "next": reverse("log_list"),
            },
        )

        self.assertRedirects(response, reverse("log_list"))
        self.log.refresh_from_db()
        self.assertEqual(self.log.temperature, Decimal("36.7"))
        self.assertIsNone(self.log.amount)
        self.assertIsNone(self.log.duration)

    def test_other_users_records_cannot_be_viewed_edited_or_deleted(self):
        other_user = get_user_model().objects.create_user(
            username="other-edit-parent",
            password="edit-test-password",
        )
        other_log = BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=timezone.now(),
            memo="private",
        )

        edit_response = self.client.get(
            reverse("log_edit", args=[other_log.pk])
        )
        edit_post_response = self.client.post(
            reverse("log_edit", args=[other_log.pk]),
            {
                "log_type": "sleep",
                "timestamp": "2026-10-20T10:30",
                "memo": "changed",
            },
        )
        delete_response = self.client.get(
            reverse("log_delete", args=[other_log.pk])
        )
        delete_post_response = self.client.post(
            reverse("log_delete", args=[other_log.pk])
        )

        self.assertEqual(edit_response.status_code, 404)
        self.assertEqual(edit_post_response.status_code, 404)
        self.assertEqual(delete_response.status_code, 404)
        self.assertEqual(delete_post_response.status_code, 404)
        other_log.refresh_from_db()
        self.assertEqual(other_log.memo, "private")
        self.assertTrue(BabyLog.objects.filter(pk=other_log.pk).exists())

    def test_daily_quick_log_post_returns_to_selected_day(self):
        daily_url = f'{reverse("daily_log")}?date=2026-10-20'

        response = self.client.post(
            daily_url,
            {
                "quick_log": "toilet",
                "next": daily_url,
            },
        )

        self.assertRedirects(response, daily_url)
        self.assertTrue(BabyLog.objects.filter(log_type="toilet").exists())

    def test_daily_detailed_post_saves_and_returns_to_selected_day(self):
        daily_url = f'{reverse("daily_log")}?date=2026-10-20'
        other_user = get_user_model().objects.create_user(
            username="spoofed-log-owner",
            password="spoof-test-password",
        )

        response = self.client.post(
            daily_url,
            {
                "log_type": "milk",
                "timestamp": "2026-10-20 10:30:00",
                "memo": "120ml",
                "user": other_user.pk,
                "next": daily_url,
            },
        )

        self.assertRedirects(response, daily_url)
        log = BabyLog.objects.get(
            user=self.user,
            timestamp=timezone.make_aware(datetime(2026, 10, 20, 10, 30)),
        )
        self.assertEqual(log.memo, "120ml")
        self.assertEqual(log.user, self.user)


class BabyLogAdminTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_superuser(
            username="admin-parent",
            email="admin-parent@example.test",
            password="admin-test-password",
        )
        self.other_user = get_user_model().objects.create_user(
            username="other-admin-parent",
            password="other-admin-test-password",
        )
        self.client.force_login(self.user)

    def test_admin_list_only_shows_the_logged_in_users_records(self):
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timezone.now(),
            memo="own admin record",
        )
        BabyLog.objects.create(
            user=self.other_user,
            log_type="milk",
            timestamp=timezone.now(),
            memo="other admin record",
        )

        response = self.client.get(reverse("admin:logs_babylog_changelist"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "own admin record")
        self.assertNotContains(response, "other admin record")

    def test_admin_form_does_not_expose_the_owner_field(self):
        response = self.client.get(reverse("admin:logs_babylog_add"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("user", response.context["adminform"].form.fields)


class AnalyticsDashboardTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="analytics-parent",
            password="test-password",
        )
        self.client.force_login(self.user)

    def test_returns_daily_milk_and_sleep_totals_for_seven_days(self):
        today = timezone.localdate()
        target_day = today - timedelta(days=2)
        morning = timezone.make_aware(
            datetime.combine(target_day, datetime.min.time()).replace(hour=8)
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=morning,
            amount=120,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=morning + timedelta(hours=2),
            amount=90,
        )
        other_user = get_user_model().objects.create_user(
            username="other-analytics-parent",
            password="analytics-test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=morning + timedelta(hours=1),
            amount=999,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=morning + timedelta(hours=3),
            duration=45,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="sleep",
            timestamp=morning + timedelta(hours=5),
            duration=30,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="temperature",
            timestamp=morning + timedelta(hours=6),
            temperature="36.5",
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="temperature",
            timestamp=morning + timedelta(hours=8),
            temperature="37.1",
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="toilet",
            timestamp=morning,
            amount=999,
            duration=999,
        )
        BabyLog.objects.create(
            user=self.user,
            log_type="milk",
            timestamp=timezone.make_aware(
                datetime.combine(today - timedelta(days=8), datetime.min.time())
            ),
            amount=500,
        )

        response = self.client.get(reverse("analytics_dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "analytics.html")
        self.assertContains(response, "直近7日間の授乳量推移（ml）")
        self.assertContains(response, "直近7日間の睡眠時間推移（時間）")
        self.assertContains(response, "直近7日間の体温推移（℃）")
        self.assertContains(response, "https://cdn.jsdelivr.net/npm/chart.js")
        self.assertContains(response, 'class="nav-link active" href="/analytics/"')
        daily_data = response.context["daily_data"]
        self.assertEqual(len(daily_data), 7)
        target_entry = next(
            item for item in daily_data if item["date"] == target_day.isoformat()
        )
        self.assertEqual(target_entry["feeding_amount_ml"], 210)
        self.assertEqual(target_entry["sleep_duration_minutes"], 75)
        self.assertEqual(target_entry["temperature_c"], 36.8)
        self.assertEqual(daily_data[-1]["date"], today.isoformat())
        self.assertEqual(daily_data[-1]["feeding_amount_ml"], 0)
        self.assertEqual(daily_data[-1]["sleep_duration_minutes"], 0)
        self.assertIsNone(daily_data[-1]["temperature_c"])
        self.assertTrue(response.context["has_feeding_data"])
        self.assertTrue(response.context["has_sleep_data"])
        self.assertTrue(response.context["has_temperature_data"])

    def test_shows_explanations_instead_of_empty_charts(self):
        response = self.client.get(reverse("analytics_dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "直近7日間に授乳量の記録はありません。")
        self.assertContains(response, "直近7日間に睡眠時間の記録はありません。")
        self.assertContains(response, "直近7日間に体温の記録はありません。")
        self.assertNotContains(response, 'id="feeding-chart"')
        self.assertNotContains(response, 'id="sleep-chart"')
        self.assertNotContains(response, 'id="temperature-chart"')

    def test_anonymous_user_is_redirected_to_login(self):
        self.client.logout()

        response = self.client.get(reverse("analytics_dashboard"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])


class DemoAccountTests(TestCase):
    def setUp(self):
        call_command("initialize_demo_accounts", verbosity=0)
        self.short_user = get_user_model().objects.get(
            username=DEMO_ACCOUNT_USERNAMES["short"]
        )
        self.long_user = get_user_model().objects.get(
            username=DEMO_ACCOUNT_USERNAMES["long"]
        )
        self.assertFalse(self.short_user.has_usable_password())
        self.assertFalse(self.long_user.has_usable_password())

    @override_settings(DEMO_ACCOUNTS_ENABLED=True)
    def test_login_page_offers_both_demo_accounts_and_posts_login(self):
        response = self.client.get(reverse("login"))

        self.assertContains(response, "デモ1：短期間のデータで試す")
        self.assertContains(response, "デモ2：長期間のデータで試す")
        response = self.client.post(
            reverse("demo_login"),
            {"demo_account": "short"},
        )

        self.assertRedirects(response, reverse("log_list"))
        self.assertEqual(
            self.client.session["_auth_user_id"],
            str(self.short_user.pk),
        )

    @override_settings(DEMO_ACCOUNTS_ENABLED=True)
    def test_long_demo_login_uses_only_the_fixed_account_mapping(self):
        response = self.client.post(
            reverse("demo_login"),
            {"demo_account": "long", "user_id": self.short_user.pk},
        )

        self.assertRedirects(response, reverse("log_list"))
        self.assertEqual(
            self.client.session["_auth_user_id"],
            str(self.long_user.pk),
        )

        self.client.logout()
        response = self.client.post(
            reverse("demo_login"),
            {"demo_account": DEMO_ACCOUNT_USERNAMES["short"]},
        )
        self.assertRedirects(response, reverse("login"))
        self.assertNotIn("_auth_user_id", self.client.session)

    @override_settings(DEMO_ACCOUNTS_ENABLED=False)
    def test_disabled_demo_feature_hides_buttons_and_rejects_server_login(self):
        response = self.client.get(reverse("login"))

        self.assertNotContains(response, "デモ1：短期間のデータで試す")
        self.assertNotContains(response, "デモ2：長期間のデータで試す")
        response = self.client.post(
            reverse("demo_login"),
            {"demo_account": "short"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_demo_seeding_is_repeatable_and_does_not_touch_regular_data(self):
        regular_user = get_user_model().objects.create_user(
            username="regular-before-demo-init",
            password="test-password",
        )
        regular_log = BabyLog.objects.create(
            user=regular_user,
            log_type="other",
            timestamp=timezone.now(),
            memo="keep this record",
        )
        initial_counts = {
            account_type: BabyLog.objects.filter(
                user=user,
                memo__startswith=f"demo-seed:v1:{account_type}:",
            ).count()
            for account_type, user in (
                ("short", self.short_user),
                ("long", self.long_user),
            )
        }

        call_command("initialize_demo_accounts", verbosity=0)

        updated_counts = {
            account_type: BabyLog.objects.filter(
                user=user,
                memo__startswith=f"demo-seed:v1:{account_type}:",
            ).count()
            for account_type, user in (
                ("short", self.short_user),
                ("long", self.long_user),
            )
        }
        self.assertEqual(initial_counts, updated_counts)
        self.assertGreaterEqual(initial_counts["short"], 15)
        self.assertGreater(initial_counts["long"], initial_counts["short"])
        self.assertTrue(BabyLog.objects.filter(pk=regular_log.pk).exists())
        self.assertEqual(regular_log.user, regular_user)

        short_types = set(
            BabyLog.objects.filter(user=self.short_user).values_list(
                "log_type", flat=True
            )
        )
        self.assertTrue(
            {"milk", "sleep", "diaper", "toilet", "temperature"} <= short_types
        )

    def test_demo_seed_has_complete_recent_days_and_comparison_data(self):
        for user, expected_days, expected_counts in (
            (
                self.short_user,
                7,
                {"milk": 6, "sleep": 6, "diaper": 6, "temperature": 1},
            ),
            (
                self.long_user,
                45,
                {"milk": 8, "sleep": 8, "diaper": 8, "temperature": 1},
            ),
        ):
            analysis = build_baby_log_analysis(
                user,
                "7d" if user == self.short_user else "30d",
            )
            self.assertGreaterEqual(
                BabyLog.objects.filter(user=user)
                .dates("timestamp", "day")
                .count(),
                expected_days,
            )
            self.assertTrue(analysis["comparisons"])
            recent_full_day = analysis["daily_data"][-2]
            for log_type, minimum_count in expected_counts.items():
                self.assertGreaterEqual(
                    recent_full_day["record_counts"][log_type],
                    minimum_count,
                )
            self.assertTrue(recent_full_day["milk_amount_complete"])
            self.assertTrue(recent_full_day["sleep_duration_complete"])

        short_analysis = build_baby_log_analysis(self.short_user, "7d")
        long_analysis = build_baby_log_analysis(self.long_user, "30d")
        self.assertTrue(short_analysis["comparisons"])
        self.assertTrue(long_analysis["comparisons"])
        self.assertGreater(short_analysis["record_count"], 100)
        self.assertGreater(long_analysis["record_count"], 700)

    @override_settings(DEMO_ACCOUNTS_ENABLED=True)
    def test_demo_profiles_show_different_data_and_read_only_controls(self):
        self.client.force_login(self.short_user)
        short_response = self.client.get(reverse("log_list"))
        short_count = short_response.context["predictions"]["feeding_log_count"]
        self.assertContains(
            short_response,
            "デモアカウントでは記録の追加・編集・削除はできません。",
        )
        self.assertNotContains(short_response, 'name="quick_log"')
        self.assertNotContains(short_response, ">編集</a>")
        self.assertNotContains(short_response, ">削除</a>")

        self.client.force_login(self.long_user)
        long_response = self.client.get(reverse("log_list"))
        long_count = long_response.context["predictions"]["feeding_log_count"]
        self.assertGreater(long_count, short_count)
        self.assertEqual(long_response.status_code, 200)

        daily_response = self.client.get(reverse("daily_log"))
        self.assertNotContains(daily_response, "quick-log-modal")
        self.assertNotContains(daily_response, 'class="fab"')

    def test_demo_accounts_cannot_add_edit_or_delete_by_direct_requests(self):
        own_log = BabyLog.objects.filter(user=self.short_user).first()
        other_user = get_user_model().objects.create_user(
            username="demo-request-other-user",
            password="test-password",
        )
        other_log = BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=timezone.now(),
            memo="must remain private",
        )
        original_count = BabyLog.objects.count()
        self.client.force_login(self.short_user)

        responses = [
            self.client.post("/", {"quick_log": "milk"}),
            self.client.post(
                reverse("daily_log"),
                {"quick_log": "milk"},
            ),
            self.client.get(reverse("log_edit", args=[own_log.pk])),
            self.client.post(
                reverse("log_edit", args=[own_log.pk]),
                {
                    "log_type": "sleep",
                    "timestamp": timezone.localtime().strftime("%Y-%m-%dT%H:%M"),
                },
            ),
            self.client.get(reverse("log_delete", args=[own_log.pk])),
            self.client.post(reverse("log_delete", args=[own_log.pk])),
            self.client.post(
                reverse("log_edit", args=[other_log.pk]),
                {"log_type": "sleep"},
            ),
            self.client.post(reverse("log_delete", args=[other_log.pk])),
        ]

        self.assertTrue(all(response.status_code == 403 for response in responses))
        self.assertEqual(BabyLog.objects.count(), original_count)
        own_log.refresh_from_db()
        other_log.refresh_from_db()
        self.assertEqual(own_log.user, self.short_user)
        self.assertEqual(other_log.memo, "must remain private")
        self.assertTrue(is_demo_account(self.short_user))
        self.assertFalse(is_demo_account(other_user))

    @override_settings(DEMO_ACCOUNTS_ENABLED=True)
    @patch(
        "logs.views.generate_baby_care_advice",
        return_value=SAMPLE_AI_REPORT,
    )
    def test_demo_can_use_analysis_prediction_and_ai_without_log_mutations(
        self, mock_generate
    ):
        other_user = get_user_model().objects.create_user(
            username="demo-ai-other-user",
            password="test-password",
        )
        BabyLog.objects.create(
            user=other_user,
            log_type="milk",
            timestamp=timezone.now(),
            amount=9999,
        )
        original_count = BabyLog.objects.count()
        self.client.force_login(self.long_user)

        home_response = self.client.get(reverse("log_list"))
        analytics_response = self.client.get(reverse("analytics_dashboard"))
        monthly_response = self.client.get(reverse("monthly_log"))
        ai_response = self.client.post(reverse("ai_advice"))

        self.assertEqual(home_response.status_code, 200)
        self.assertEqual(analytics_response.status_code, 200)
        self.assertEqual(monthly_response.status_code, 200)
        self.assertEqual(ai_response.status_code, 200)
        self.assertContains(ai_response, "記録のまとまりを確認しました。")
        mock_generate.assert_called_once()
        self.assertNotIn("9999", mock_generate.call_args.args[0])
        self.assertEqual(BabyLog.objects.count(), original_count)

    def test_demo_login_requires_post_and_csrf(self):
        response = self.client.get(reverse("demo_login"))
        self.assertEqual(response.status_code, 403)

        csrf_client = Client(enforce_csrf_checks=True)
        response = csrf_client.post(
            reverse("demo_login"),
            {"demo_account": "short"},
        )
        self.assertEqual(response.status_code, 403)


class DemoAccountSeedCollisionTests(TestCase):
    def test_seed_command_refuses_to_claim_an_existing_regular_username(self):
        regular_user = get_user_model().objects.create_user(
            username=DEMO_ACCOUNT_USERNAMES["short"],
            password="test-password",
        )

        with self.assertRaises(CommandError):
            call_command("initialize_demo_accounts", verbosity=0)

        self.assertFalse(is_demo_account(regular_user))
        self.assertFalse(
            Group.objects.filter(
                name=DEMO_ACCOUNT_GROUP,
                user__username=DEMO_ACCOUNT_USERNAMES["short"],
            ).exists()
        )
