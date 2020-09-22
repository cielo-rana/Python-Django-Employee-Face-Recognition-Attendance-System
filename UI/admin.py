from django.contrib import admin
from . models import Profile, Employee, Attendance

# Register your models here.

 
class ProfAdmin(admin.ModelAdmin):
    list_display = ('name','desc','img')

admin.site.register(Profile, ProfAdmin)  

class EmpAdmin(admin.ModelAdmin):
    list_display = ('name','phone')

admin.site.register(Employee, EmpAdmin)

class AtAdmin(admin.ModelAdmin):
    list_display = ('name','date')

admin.site.register(Attendance, AtAdmin)

admin.site.site_header = "Miracle Admin"