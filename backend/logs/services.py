from datetime import timedelta

from django.utils import timezone

from .models import BabyLog


DEFAULT_FEEDING_INTERVAL = timedelta(hours=3)
DEFAULT_WAKE_WINDOW = timedelta(hours=2, minutes=30)
FEEDING_LOG_TYPES = ("milk", "feeding")
PREDICTION_REFERENCE_PERIODS = (7, 14, 30, 35, 60)
DEFAULT_PREDICTION_REFERENCE_DAYS = 7


def predict_next_events(user, reference_days=DEFAULT_PREDICTION_REFERENCE_DAYS):
    """Predict events from one user's logs in the selected reference period."""
    if (
        isinstance(reference_days, bool)
        or not isinstance(reference_days, int)
        or reference_days not in PREDICTION_REFERENCE_PERIODS
    ):
        raise ValueError(
            "reference_days must be one of "
            f"{PREDICTION_REFERENCE_PERIODS}."
        )

    now = timezone.now()
    reference_start = now - timedelta(days=reference_days)
    feeding_logs = list(
        BabyLog.objects.filter(
            user=user,
            log_type__in=FEEDING_LOG_TYPES,
            timestamp__gte=reference_start,
            timestamp__lte=now,
        ).order_by("timestamp")
    )

    if len(feeding_logs) >= 2:
        intervals = [
            later.timestamp - earlier.timestamp
            for earlier, later in zip(feeding_logs, feeding_logs[1:])
            if later.timestamp > earlier.timestamp
        ]
    else:
        intervals = []

    average_feeding_interval = (
        sum(intervals, timedelta()) / len(intervals)
        if intervals
        else DEFAULT_FEEDING_INTERVAL
    )
    last_feeding = feeding_logs[-1].timestamp if feeding_logs else now

    sleep_logs = list(
        BabyLog.objects.filter(
            user=user,
            log_type="sleep",
            timestamp__gte=reference_start,
            timestamp__lte=now,
        )
        .order_by("timestamp")
    )
    if sleep_logs:
        awake_intervals = []
        for previous_sleep, next_sleep in zip(sleep_logs, sleep_logs[1:]):
            previous_wake_time = previous_sleep.timestamp
            if previous_sleep.duration is not None:
                previous_wake_time += timedelta(minutes=previous_sleep.duration)
            awake_interval = next_sleep.timestamp - previous_wake_time
            if awake_interval > timedelta():
                awake_intervals.append(awake_interval)

        average_wake_window = (
            sum(awake_intervals, timedelta()) / len(awake_intervals)
            if awake_intervals
            else DEFAULT_WAKE_WINDOW
        )
        last_wake_time = sleep_logs[-1].timestamp
        if sleep_logs[-1].duration is not None:
            last_wake_time += timedelta(minutes=sleep_logs[-1].duration)
    else:
        last_wake_time = now
        average_wake_window = DEFAULT_WAKE_WINDOW

    return {
        "next_feeding": last_feeding + average_feeding_interval,
        "next_sleep": last_wake_time + average_wake_window,
        "reference_days": reference_days,
        "feeding_log_count": len(feeding_logs),
        "feeding_interval_count": len(intervals),
        "feeding_data_sufficient": bool(intervals),
        "feeding_method": (
            "observed_average" if intervals else "standard_interval"
        ),
        "sleep_log_count": len(sleep_logs),
        "sleep_interval_count": len(awake_intervals) if sleep_logs else 0,
        "sleep_data_sufficient": bool(sleep_logs) and bool(awake_intervals),
        "sleep_method": (
            "observed_average"
            if sleep_logs and awake_intervals
            else "standard_wake_window"
        ),
    }
