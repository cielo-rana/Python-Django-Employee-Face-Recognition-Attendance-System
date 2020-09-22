# Generated by Django 2.2.12 on 2020-09-14 10:41

from django.db import migrations
import phone_field.models


class Migration(migrations.Migration):

    dependencies = [
        ('UI', '0002_employee'),
    ]

    operations = [
        migrations.AlterField(
            model_name='employee',
            name='phone',
            field=phone_field.models.PhoneField(blank=True, max_length=31, unique=True),
        ),
    ]
