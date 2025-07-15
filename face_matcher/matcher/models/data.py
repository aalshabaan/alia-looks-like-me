from django.contrib.auth.models import User
from django.db import models


class FaceMatcherUser(User):
    pass


class Embedding(models.Model):
    user = models.ForeignKey(
        to=FaceMatcherUser,
        verbose_name='Embedding',
        related_name='embeddings',
        on_delete=models.CASCADE
    )


class Label(models.Model):
    ordinal = models.IntegerField(
        verbose_name='Numeric Label'
    )

    label = models.CharField(
        max_len=20,
        verbose_name='Label',
        blank=False
    )

    user = models.ForeignKey(
        to=FaceMatcherUser,
        on_delete=models.CASCADE,
        related_name='labels'
    )

