# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-05-15 08:35
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('RecSite', '0008_delete_buyrecord'),
    ]

    operations = [
        migrations.CreateModel(
            name='BuyRecord',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('userId', models.CharField(max_length=10)),
                ('itemIds', models.TextField()),
            ],
        ),
    ]
