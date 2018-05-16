from django.db import models


class TopK(models.Model):
    userId = models.CharField(max_length=10)
    topK = models.TextField()

    def __str__(self):
        return self.topK


class Item(models.Model):
    itemId = models.CharField(max_length=100)
    name = models.CharField(max_length=200)
    price = models.CharField(max_length=100)
    type = models.CharField(max_length=100)

    def __str__(self):
        return self.itemId


class BuyRecord(models.Model):
    userId = models.CharField(max_length=10)
    itemIds = models.TextField()

    def __str__(self):
        return self.itemIds

