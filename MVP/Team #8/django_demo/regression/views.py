from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseServerError
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import json

from django.views.decorators.http import require_http_methods
from django.views.decorators.clickjacking import xframe_options_sameorigin, xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt

import os
import redis
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

def get_redis_connection():
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=0, 
            decode_responses=True,
            socket_connect_timeout=5,  # 5 second timeout
            socket_timeout=5
        )
        redis_client.ping()  # Test connection
        return redis_client
    except redis.ConnectionError as e:
        print(f"Redis Connection Error: {e}")
        raise


def index(request):
    return render(request, 'regression/index.html')

def regression(request):
    
    DATASET_DIR = os.getenv('DATASET_DIR', '/app/datasets')
    
    available_datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    
    context = {
        'datasets': available_datasets
    }
    
    logger.debug(f"Available datasets: {available_datasets}")
    
    return render(request, 'pages/regression.html', context=context)

def compute_regression(request):
    if request.method == 'POST':
        data = request.POST.getlist('data[]')
        data = np.array([list(map(float, point.split(','))) for point in data])
        X, y = data[:, :-1], data[:, -1]

        model = LinearRegression()
        model.fit(X, y)
        
        intercept = model.intercept_
        coefficients = model.coef_
        
        return JsonResponse({'intercept': intercept, 'coefficients': list(coefficients)})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def compute_regression(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        data = body.get('data')
        data = np.array([list(map(float, point.split(','))) for point in data])
        X, y = data[:, :-1], data[:, -1]

        model = LinearRegression()
        model.fit(X, y)
        
        intercept = model.intercept_
        coefficients = model.coef_
        
        return JsonResponse({'intercept': intercept, 'coefficients': list(coefficients)})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@require_http_methods(["POST"])
def redis_post(request, app, key):
    
    try:
        # Get data from the HTMX post request
        value = request.POST.get('value')
        
        if not key or not value:
            return JsonResponse({
                'status': 'error', 
                'message': 'Key and value are required'
            }, status=400)
        
        # Store in Redis
        redis_client = get_redis_connection()
        
        key = f'{app}:{key}'
        
        redis_client.set(key, value)
        
        return HttpResponse(value)
        
        return JsonResponse({
            'status': 'success', 
            'message': f'Stored {key}: {value} in Redis'
        })
    
    except Exception as e:
        return JsonResponse({
            'status': 'error', 
            'message': str(e)
        }, status=500)
