from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import redis

# Create Redis client at app initialization
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

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
except redis.ConnectionError as e:
    print(f"Redis Connection Error: {e}")
    redis_client = None

app_name = "linear_regression"
app = Dash(app_name)
   
app.layout = html.Div([
    dcc.Interval(id='update-interval', interval=50, n_intervals=0),  # Poll Redis every 2 seconds
    dcc.Graph(id='linear-graph')
])

@app.callback(
    Output('linear-graph', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_graph(n_intervals):
    if redis_client is None:
        # Handle the case where Redis connection failed
        fig = go.Figure()
        fig.update_layout(title="Redis Connection Error")
        return fig
   
    # Fetch parameters from Redis
    a_key = f'{app_name}:a'
    b_key = f'{app_name}:b'
    a = float(redis_client.get(a_key) or 1)  # Default to 1 if key doesn't exist
    b = float(redis_client.get(b_key) or 0)  # Default to 0 if key doesn't exist
    
    # Generate graph
    x = np.linspace(-10, 10, 100)
    y = a * x + b
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig

if __name__ == '__main__':
    PORT = int(os.environ.get('DASH_PORT', 8051))
    DEBUG = int(os.environ.get('DASH_DEBUG', 0))
    app.run_server(host='0.0.0.0', debug=DEBUG, port=PORT)