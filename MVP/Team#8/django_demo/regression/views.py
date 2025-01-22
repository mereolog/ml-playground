from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from sklearn.linear_model import LinearRegression

def index(request):
    return render(request, 'regression/index.html')

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

# regression/views.py
import json
from django.views.decorators.csrf import csrf_exempt

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

