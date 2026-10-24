from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def assign_existing_logs_to_admin(apps, schema_editor):
    database = schema_editor.connection.alias
    BabyLog = apps.get_model("logs", "BabyLog")
    User = apps.get_model(*settings.AUTH_USER_MODEL.split("."))
    unowned_logs = BabyLog.objects.using(database).filter(user__isnull=True)

    if not unowned_logs.exists():
        return

    try:
        owner = User.objects.using(database).get(username="admin")
    except User.DoesNotExist:
        raise RuntimeError(
            "Existing BabyLog records require the confirmed admin user."
        ) from None

    unowned_logs.update(user_id=owner.pk)


class Migration(migrations.Migration):

    dependencies = [
        ("logs", "0004_alter_babylog_amount"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name="babylog",
            name="user",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="baby_logs",
                to=settings.AUTH_USER_MODEL,
                verbose_name="ユーザー",
            ),
        ),
        migrations.RunPython(
            assign_existing_logs_to_admin,
            migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name="babylog",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.PROTECT,
                related_name="baby_logs",
                to=settings.AUTH_USER_MODEL,
                verbose_name="ユーザー",
            ),
        ),
    ]
