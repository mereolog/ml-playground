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
