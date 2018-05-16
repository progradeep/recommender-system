from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
# Register your models here.
from .models import *


@admin.register(Item)
class Item(ImportExportModelAdmin):
    pass

@admin.register(TopK)
class TopK(ImportExportModelAdmin):
    pass

@admin.register(BuyRecord)
class BuyRecord(ImportExportModelAdmin):
    pass