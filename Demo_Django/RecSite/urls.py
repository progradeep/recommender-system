from django.conf.urls import url, include
from django.contrib import admin
from RecSite.views import *

# / 뒤에 어떤 url이 붙느냐에 따라 어떤 함수를 부를지 적어놓은 파일

urlpatterns = [
    url(r'^$', main, name= 'main'),
    url(r'^search$',show_results, name='show_results')
]