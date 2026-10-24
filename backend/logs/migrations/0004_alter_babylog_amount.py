from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logs', '0003_babylog_amount_duration_temperature'),
    ]

    operations = [
        migrations.AlterField(
            model_name='babylog',
            name='amount',
            field=models.IntegerField(
                blank=True,
                null=True,
                verbose_name='量（ml）',
            ),
        ),
    ]
