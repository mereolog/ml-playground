from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.http import FileResponse

import requests
import json
import os

import logging

logger = logging.getLogger(__name__)

@csrf_exempt
@csrf_protect
@require_POST
def load_file(request):
    data = json.loads(request.body)
    file = data.get('datasetFileName')

    DATASET_DIR = os.getenv('DATASET_DIR', '/app/datasets')
    
    try:
        file_path = os.path.join(DATASET_DIR, file)
        logger.debug(f"File path: {file_path}")
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=file)
        else:
            return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def send_data(request):
    if request.method == "POST":
        # Step 1: Handle file upload (if any)
        dataset_file = request.FILES.get('dataset-file')
        if not dataset_file:
            return JsonResponse({"error": "No file uploaded."})
        else:
            dataset_content = dataset_file.read().decode('utf-8')
        # Step 2: Capture other form data
        dependent_variable = request.POST.get('dependent-variable')
        independent_variable = request.POST.get('independent-variable')
        learning_rate = request.POST.get('learning-rate')
        epochs = request.POST.get('epochs')
        batch_size = request.POST.get('batch-size')
        regularization = request.POST.get('regularization')

        # Step 3: Process data if needed (e.g., convert to float, validate, etc.)
        try:
            learning_rate = float(learning_rate)
            epochs = int(epochs)
            batch_size = int(batch_size)
        except ValueError as e:
            return JsonResponse({"error": f"Invalid input: {str(e)}"}, status=400)

        # Step 4: Here you can use the form data to process or train the model
        # For now, we'll just return the form data as a response
        data = {
            "dependent_variable": dependent_variable,
            "independent_variable": independent_variable,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "regularization": regularization,
            "csv_data": dataset_content,
        }

        # Step 5: Send the POST request to FastAPI
        fastapi_url = "http://fastapi:8000/process-data"  # Replace with your FastAPI app URL
        response = requests.post(fastapi_url, json=data)

        # Step 6: Handle the response from FastAPI
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({"error": "Failed to process data in FastAPI."}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)