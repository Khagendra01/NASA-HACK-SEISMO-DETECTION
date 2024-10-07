
from django.urls import path
from .views import Get_ResponseView, Get_Predict

urlpatterns = [
    path('get-ai-response/', Get_ResponseView.as_view(), name='ai-response'),
    path('predict/', Get_Predict.as_view(), name='predict'),
]
