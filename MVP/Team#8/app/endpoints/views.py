import json
import os
import logging

from django.http import JsonResponse, FileResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def load_file(request):
    data = json.loads(request.body)
    file = data.get("datasetFileName")
    dataset_dir = os.getenv("dataset_dir", "/app/datasets")
    try:
        file_path = os.path.join(dataset_dir, file)
        logger.debug("File path: %s", file_path)

        if os.path.exists(file_path):
            return FileResponse(open(file_path, "rb"), as_attachment=True, filename=file)
        return JsonResponse({"error": "File not found"}, status=404)

    except (OSError, ValueError) as e:
        logger.error("Error while loading file: %s", e)
        return JsonResponse({"error": str(e)}, status=500)