from django.conf import settings
from django.db import models


class BabyLog(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="baby_logs",
        verbose_name="ユーザー",
    )

    # 記録の種類（授乳、睡眠、おむつなど）
    LOG_TYPES = [
        ('milk', '授乳'),
        ('sleep', '睡眠'),
        ('diaper', 'おむつ交換'),
        ('toilet', '排泄'),
        ('temperature', '体温'),
        ('other', 'その他'),
    ]

    log_type = models.CharField(max_length=20, choices=LOG_TYPES, verbose_name="記録タイプ")
    timestamp = models.DateTimeField(verbose_name="日時")
    amount = models.IntegerField(
        blank=True,
        null=True,
        verbose_name="量（ml）",
    )
    duration = models.PositiveIntegerField(
        blank=True,
        null=True,
        verbose_name="時間（分）",
    )
    temperature = models.DecimalField(
        max_digits=4,
        decimal_places=1,
        blank=True,
        null=True,
        verbose_name="体温（℃）",
    )
    memo = models.TextField(blank=True, null=True, verbose_name="メモ")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="作成日時")

    def __str__(self):
        return f"{self.get_log_type_display()} {self.timestamp.strftime('%Y-%m-%d %H:%M')}"