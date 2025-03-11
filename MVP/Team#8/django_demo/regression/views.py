import os
import json
import logging
import numpy as np

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from sklearn.linear_model import LinearRegression

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
            socket_connect_timeout=5,  # 5-second timeout
            socket_timeout=5
        )
        redis_client.ping()  # Test connection
        return redis_client
    except redis.ConnectionError as e:
        logger.error("Redis Connection Error: %s", e)
        raise


def index(request):
    return render(request, 'regression/index.html')


def regression(request):
    dataset_dir = os.getenv('DATASET_DIR', '/app/datasets')

    available_datasets = [
        f for f in os.listdir(dataset_dir) if f.endswith('.csv')
    ]

    context = {'datasets': available_datasets}

    logger.debug("Available datasets: %s", available_datasets)

    return render(request, 'pages/regression.html', context=context)


@csrf_exempt
def compute_regression(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        data = body.get('data')

        if not data:
            return JsonResponse({'error': 'No data provided'}, status=400)

        try:
            data_array = np.array([list(map(float, point.split(','))) for point in data])
            x_values, y_values = data_array[:, :-1], data_array[:, -1]

            model = LinearRegression()
            model.fit(x_values, y_values)

            intercept = model.intercept_
            coefficients = model.coef_

            return JsonResponse({'intercept': intercept, 'coefficients': list(coefficients)})
        except ValueError as e:
            logger.error("Error processing regression data: %s", e)
            return JsonResponse({'error': 'Invalid data format'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)


@require_http_methods(["POST"])
def redis_post(request, app, key):
    try:
        value = request.POST.get('value')

        if not key or not value:
            return JsonResponse({'status': 'error', 'message': 'Key and value are required'},
                                status=400)

        redis_client = get_redis_connection()
        redis_key = f'{app}:{key}'

        redis_client.set(redis_key, value)

        return JsonResponse({
            'status': 'success',
            'message': f'Stored {redis_key}: {value} in Redis'})

    except Exception as e:
        logger.error("Redis error: %s", e)
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
