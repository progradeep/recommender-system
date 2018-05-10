from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
# Register your models here.
from .models import *


@admin.register(shopDB)
class ShopDBAdmin(ImportExportModelAdmin):
    pass

@admin.register(TopK)
class ShopDBAdmin(ImportExportModelAdmin):
    pass