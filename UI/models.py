from django.db import models
from phone_field import PhoneField

# Create your models here.
class Profile(models.Model):
    
    name = models.CharField(max_length=100)
    desc = models.CharField(max_length=100)
    img = models.ImageField(upload_to='img')


class Employee(models.Model):

    name = models.CharField(max_length=100)
    phone = PhoneField(blank=True, null=False, unique=True)

class Attendance(models.Model):

    name = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now=True)  