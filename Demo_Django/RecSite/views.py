from django.shortcuts import render, redirect
from .models import *

# url에 따른 함수 정의해놓은 파일
# 함수를 부를때 클라이언트의 정보 (ip / 시간 등)  => request

# Create your views here.
def main(request):
    userId = request.GET.get('q')

    if userId:
        topK_itemIds = TopK.objects.get(userId=userId)
        topK_itemIds = str(topK_itemIds).split(",")

        product = []

        for itemId in topK_itemIds:
            p = shopDB.objects.get(product_id=itemId)
            product.append({
                'product_id': p.product_id,
                'name': p.clothes,
                'price': p.price,
                'type': p.type, })

        context = {'query':userId,
                   'items': product}

        return render(request, 'RecSite/searchList.html', context)

    else:
        return render(request,"RecSite/main.html")

# def accept_list(request):

# def receiver(request):
#     userId = request.GET.get('q')
#
#     topK_itemIds = TopK.objects.get(userId=userId)
#     topK_itemIds = topK_itemIds.split(",")
#
#     product = []
#
#     for itemId in topK_itemIds:
#         p = shopDB.objects.get(product_id=itemId)
#         product.append({
#             'product_id': p.product_id,
#             'name': p.clothes,
#             'price': p.price,
#             'type': p.type, })
#         print(p.clothes)
#
#     context = {'items':product}
#
#     return render(request,'RecSite/main.html', context)