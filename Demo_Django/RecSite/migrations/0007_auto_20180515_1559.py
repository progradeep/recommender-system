# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-05-15 06:59
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('RecSite', '0006_auto_20180515_1521'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='shopDB',
            new_name='Item',
        ),
    ]
