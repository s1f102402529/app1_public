from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logs', '0002_babylog_log_type_toilet_temperature'),
    ]

    operations = [
        migrations.AddField(
            model_name='babylog',
            name='amount',
            field=models.DecimalField(
                blank=True,
                decimal_places=1,
                max_digits=7,
                null=True,
                verbose_name='量（ml）',
            ),
        ),
        migrations.AddField(
            model_name='babylog',
            name='duration',
            field=models.PositiveIntegerField(
                blank=True,
                null=True,
                verbose_name='時間（分）',
            ),
        ),
        migrations.AddField(
            model_name='babylog',
            name='temperature',
            field=models.DecimalField(
                blank=True,
                decimal_places=1,
                max_digits=4,
                null=True,
                verbose_name='体温（℃）',
            ),
        ),
    ]
