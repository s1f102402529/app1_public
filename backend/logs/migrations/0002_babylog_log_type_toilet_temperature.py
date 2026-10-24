from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logs', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='babylog',
            name='log_type',
            field=models.CharField(
                choices=[
                    ('milk', '授乳'),
                    ('sleep', '睡眠'),
                    ('diaper', 'おむつ交換'),
                    ('toilet', '排泄'),
                    ('temperature', '体温'),
                    ('other', 'その他'),
                ],
                max_length=20,
                verbose_name='記録タイプ',
            ),
        ),
    ]
