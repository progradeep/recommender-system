from django.shortcuts import render, redirect
from .models import *
import random, pprint

# url에 따른 함수 정의해놓은 파일
# 함수를 부를때 클라이언트의 정보 (ip / 시간 등)  => request

# Create your views here.
def main(request):
    return render(request, 'RecSite/main.html', {})


def show_results(request):
    ########### Changed! No more getting userId! Just show them all!
    userIds = list(BuyRecord.objects.values_list('userId', flat=True))

    random.shuffle(userIds)

    userIds = userIds[:50]

    context = {}
    context['context'] = {}

    for userId in userIds:
        try:
            buy_itemIds = BuyRecord.objects.get(userId=userId)
            buy_itemIds = str(buy_itemIds).split(",")
            buy_itemIds = [x for x in buy_itemIds if x != '']

            if len(buy_itemIds) < 3:
                continue
            else:
                buy_itemIds = buy_itemIds[:3]

        except:
            continue

        try:
            topK_itemIds = TopK.objects.get(userId=userId)
            topK_itemIds = str(topK_itemIds).split(",")


        except:
            continue

        results = []

        pur_items = [] # check for item duplication
        for itemId in buy_itemIds:
            p = Item.objects.get(itemId=itemId)
            pname = p.name

            if pname in pur_items:
                continue

            results.append({
                'itemId': p.itemId[1:],
                'name': p.name,
                'price': p.price,
                'type': p.type,
            })

            pur_items.append(pname)

            if len(results) == 3:
                break


        rec_items = []
        for itemId in topK_itemIds:
            p = Item.objects.get(itemId=itemId)
            pname = p.name

            if pname in pur_items or pname in rec_items:
                continue

            p = Item.objects.get(itemId=itemId)

            results.append({
                'itemId': p.itemId[1:],
                'name': p.name,
                'price': p.price,
                'type': p.type, })

            rec_items.append(itemId)

            if len(results) == 6:
                break

        context['context'][userId] = results
        context['totalLength'] = len(userIds)

    # pprint.pprint(context)

    return render(request, 'RecSite/searchList.html', context)


