from django.urls import path
from . import views


urlpatterns = [
    path('', views.UI, name='UI'),
    path('developer/', views.sec, name='sec'),
    path('attendance/', views.fou, name='fou'),
    path('register/', views.reg, name='reg'),
    path('login/', views.login, name='login'),
    path('logout',views.logout,name='logout'),
]
