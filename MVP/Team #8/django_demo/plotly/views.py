from django.http import JsonResponse
import json
import numpy as np

def update_plot(request, plot='', key=''):
    # Define the new data for the plot
    
    a = float(request.POST.get('a', 1.0))
    b = float(request.POST.get('b', 1.0))
    
    
    x = np.linspace(-10, 10, 198, int)

    y = a * x + b
    
    plot_data = [
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "mode": "lines",
            "type": "scatter"
        }
    ]

    # Optional: Define layout updates (if needed)
    layout = {"title": "Updated Plot"}

    # Create the response object
    response_data = {
        "restyle_data": json.dumps(plot_data),  # Serialize plot data
        "layout": json.dumps(layout),  # Serialize layout data
    }

    return JsonResponse(response_data)