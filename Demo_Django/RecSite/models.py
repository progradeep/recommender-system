from django.db import models


class TopK(models.Model):
    userId = models.CharField(max_length=10)
    topK = models.TextField()

    def __str__(self):
        return self.topK


class shopDB(models.Model):
    product_id = models.CharField(max_length=100)
    clothes = models.CharField(max_length=200)
    price = models.CharField(max_length=100)
    type = models.CharField(max_length=100)

    def __str__(self):
        return self.product_id