import json
import os

import requests
from django.http import HttpResponse
from django.shortcuts import render

FASTAPI_URL = "http://fastapi:8000"


def home_view(request):
    available_algorithms = requests.get(f"{FASTAPI_URL}/algorithms").json()
    print(available_algorithms)

    dataset_dir = os.getenv("DATASET_DIR", "/app/datasets")

    available_datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    context = {"algorithms": available_algorithms, "datasets": available_datasets}

    return render(request, "pages/playground.html", context=context)


def linear_regression(request):

    return render(request, "algorithms/linear_regression.html", context=context)


def algorithm_form(request, algorithm_name):
    resp = requests.get(f"{FASTAPI_URL}/algorithms/{algorithm_name}/config_schema")
    if resp.status_code != 200:
        return HttpResponse("Could not fetch schema", status=500)
    schema = resp.json()
    print(schema)
    return render(request, "partials/algorithm_inputs.html", {"schema": schema})

