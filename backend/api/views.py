from django.shortcuts import render
from .chat import get_ai_response
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

class Get_ResponseView(APIView):
    def post(self, request):  
        question = request.data.get('question', None)
        past_convo = request.data.get('past_convo', None)
        response = get_ai_response(question, past_convo)

        return Response({'response': response}, status=status.HTTP_200_OK)