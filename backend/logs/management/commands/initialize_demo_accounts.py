from datetime import datetime, time, timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from logs.demo_accounts import (
    DEMO_ACCOUNT_GROUP,
    DEMO_ACCOUNT_USERNAMES,
    DEMO_SEED_PREFIX,
)
from logs.models import BabyLog


def _scheduled_event(log_type, hour, minute=0, **values):
    return {
        "log_type": log_type,
        "hour": hour,
        "minute": minute,
        "amount": values.get("amount"),
        "duration": values.get("duration"),
        "temperature": values.get("temperature"),
    }


def _events_for_day(account_type, days_ago):
    if account_type == "short":
        recentness = 6 - days_ago
        events = [
            _scheduled_event(
                "milk",
                hour,
                amount=145 + (recentness // 3) * 5 + index % 2 * 5,
            )
            for index, hour in enumerate((2, 6, 10, 14, 18, 22))
        ]
        events.extend(
            _scheduled_event("sleep", hour, duration=duration)
            for hour, duration in (
                (0, 90),
                (4, 90),
                (8, 90),
                (12, 90),
                (16, 75),
                (20, 105),
            )
        )
        events.extend(
            _scheduled_event("diaper", hour)
            for hour in (1, 5, 9, 13, 17, 21)
        )
        events.append(_scheduled_event("toilet", 11, 30))
        events.append(
            _scheduled_event(
                "temperature",
                7,
                30,
                temperature=Decimal("36.5") + Decimal(recentness % 3) / 10,
            )
        )
        if days_ago % 2 == 0:
            events.append(_scheduled_event("toilet", 18, 30))
        return events

    recentness = 44 - days_ago
    events = [
        _scheduled_event(
            "milk",
            hour,
            minute=15,
            amount=140 + (recentness // 15) * 5 + index % 3 * 5,
        )
        for index, hour in enumerate((2, 5, 8, 11, 14, 17, 20, 23))
    ]
    events.extend(
        _scheduled_event("sleep", hour, duration=duration)
        for hour, duration in (
            (0, 120),
            (3, 90),
            (7, 90),
            (10, 90),
            (13, 90),
            (16, 60),
            (19, 90),
            (22, 120),
        )
    )
    events.extend(
        _scheduled_event("diaper", hour, minute=30)
        for hour in (1, 4, 7, 10, 13, 16, 19, 22)
    )
    events.extend(
        (
            _scheduled_event("toilet", 9, 30),
            _scheduled_event("toilet", 17, 30),
            _scheduled_event(
                "temperature",
                7,
                45,
                temperature=Decimal("36.4") + Decimal(recentness % 4) / 10,
            ),
        )
    )
    if days_ago % 3 == 0:
        events.append(
            _scheduled_event(
                "temperature",
                19,
                15,
                temperature=Decimal("36.5") + Decimal(recentness % 3) / 10,
            )
        )
    if days_ago % 2 == 0:
        events.append(_scheduled_event("other", 12, 45))
    return events


def _seed_rows(account_type, days, now):
    current_timezone = timezone.get_current_timezone()
    today = timezone.localdate(now)
    rows = {}

    for days_ago in range(days):
        event_date = today - timedelta(days=days_ago)
        for event_index, event in enumerate(
            _events_for_day(account_type, days_ago), start=1
        ):
            timestamp = timezone.make_aware(
                datetime.combine(
                    event_date,
                    time(event["hour"], event["minute"]),
                ),
                current_timezone,
            )
            if timestamp > now:
                continue

            marker = (
                f"{DEMO_SEED_PREFIX}{account_type}:"
                f"{event_date:%Y%m%d}:{event_index}"
            )
            rows[marker] = {
                "log_type": event["log_type"],
                "timestamp": timestamp,
                "amount": event["amount"],
                "duration": event["duration"],
                "temperature": event["temperature"],
                "memo": marker,
            }
    return rows


class Command(BaseCommand):
    help = "Create or refresh the two isolated, read-only demo accounts and sample logs."

    @transaction.atomic
    def handle(self, *args, **options):
        user_model = get_user_model()
        group, _ = Group.objects.get_or_create(name=DEMO_ACCOUNT_GROUP)
        now = timezone.now()

        for account_type, username in DEMO_ACCOUNT_USERNAMES.items():
            user, created = user_model.objects.get_or_create(
                username=username,
                defaults={"is_active": True},
            )
            if not created and not user.groups.filter(pk=group.pk).exists():
                raise CommandError(
                    f"Refusing to claim existing non-demo user '{username}'. "
                    "Review the account before assigning it to the demo group."
                )

            user = user_model.objects.select_for_update().get(pk=user.pk)
            user.groups.add(group)
            if user.is_staff or user.is_superuser or not user.is_active:
                user.is_staff = False
                user.is_superuser = False
                user.is_active = True
                user.save(
                    update_fields=("is_staff", "is_superuser", "is_active")
                )
            if created or user.has_usable_password():
                user.set_unusable_password()
                user.save(update_fields=("password",))

            days = 7 if account_type == "short" else 45
            seed_rows = _seed_rows(account_type, days, now)
            prefix = f"{DEMO_SEED_PREFIX}{account_type}:"
            existing = list(
                BabyLog.objects.filter(
                    user=user,
                    memo__startswith=prefix,
                ).order_by("pk")
            )
            existing_by_marker = {}
            for log in existing:
                if log.memo in seed_rows and log.memo not in existing_by_marker:
                    existing_by_marker[log.memo] = log
                else:
                    log.delete()

            for marker, values in seed_rows.items():
                log = existing_by_marker.get(marker)
                if log is None:
                    BabyLog.objects.create(user=user, **values)
                    continue
                for field, value in values.items():
                    setattr(log, field, value)
                log.save(
                    update_fields=(
                        "log_type",
                        "timestamp",
                        "amount",
                        "duration",
                        "temperature",
                        "memo",
                    )
                )

            self.stdout.write(
                self.style.SUCCESS(
                    f"{username}: {len(seed_rows)} demo records initialized "
                    f"({days} days)."
                )
            )
