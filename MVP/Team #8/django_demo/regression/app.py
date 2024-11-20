from dash import Dash, html, dcc, callback, Output, Input
import dash
# import dash
from flask import Flask
import plotly.express as px
import pandas as pd
import os

from django_plotly_dash import DjangoDash

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = DjangoDash('dash_regression')

app.layout = html.Div([
    html.H1("Interactive Linear Function Visualizer", style={'textAlign': 'center'}),
    
    # Sliders for 'a' and 'b'
    html.Div([
        html.Label("Slope (a):"),
        dcc.Slider(
            id='slope-slider',
            min=-10, max=10, step=0.1, value=1,
            marks={i: str(i) for i in range(-10, 11)}
        ),
        html.Label("Intercept (b):"),
        dcc.Slider(
            id='intercept-slider',
            min=-20, max=20, step=0.1, value=0,
            marks={i: str(i) for i in range(-20, 21, 5)}
        ),
    ], style={'width': '50%', 'margin': 'auto'}),
    
    # Graph
    dcc.Graph(id='linear-graph', style={'height': '70vh'}),
])

# Callback to update the graph
@app.callback(
    Output('linear-graph', 'figure'),
    Input('slope-slider', 'value'),
    Input('intercept-slider', 'value')
)
def update_graph(a, b):
    # Generate x and y data
    x = np.linspace(-10, 10, 500)
    y = a * x + b
    
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"y = {a}x + {b}"))
    fig.update_layout(
        title=f"Linear Function: y = {a}x + {b}",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_dark",
    )
    return fig