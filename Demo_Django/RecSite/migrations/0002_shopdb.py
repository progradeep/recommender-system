# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-05-03 15:01
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RecSite', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='shopDB',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_id', models.CharField(max_length=100)),
                ('clothes', models.CharField(max_length=200)),
                ('price', models.CharField(max_length=100)),
                ('type', models.CharField(max_length=100)),
            ],
        ),
    ]
