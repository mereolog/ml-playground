import os

from django.shortcuts import render


def linear_regression(request):
    dataset_dir = os.getenv("DATASET_DIR", "/app/datasets")

    available_datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    context = {"datasets": available_datasets}

    return render(request, "algorithms/linear_regression.html", context=context)
