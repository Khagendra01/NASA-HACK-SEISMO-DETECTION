
from django.urls import path
from .views import Get_ResponseView

urlpatterns = [
    path('get-ai-response/', Get_ResponseView.as_view(), name='ai-response'),
]
