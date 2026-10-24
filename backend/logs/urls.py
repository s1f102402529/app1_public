# backend/logs/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.log_list, name='log_list'),
    path('daily/', views.daily_log, name='daily_log'),
    path('monthly/', views.monthly_log, name='monthly_log'),
    path('logs/<int:pk>/edit/', views.log_edit, name='log_edit'),
    path('logs/<int:pk>/delete/', views.log_delete, name='log_delete'),
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    path('advice/', views.ai_advice, name='ai_advice'),
]