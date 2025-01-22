from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global variables for model state



data = None
model = None
X = None
y = None
costs = []
current_epoch = 0
max_epochs = 10
learning_rate = 0.01
weights = None
bias = None


@app.route('/')
# Route for rendering the home page
@app.route('/')
def index():
    """
    Render the home page of the application.
    """
    return render_template('index.html')

# Route for loading the dataset
@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    global data, X, y, costs, current_epoch, weights, bias
    file = request.files.get('file')
    if file:
        try:
            data = pd.read_csv(file)
            costs = []
            current_epoch = 0
            weights = None
            bias = None
            # Check for required columns
            if 'YearsExperience' not in data.columns or 'Salary' not in data.columns:
                return jsonify({"error": "Dataset must contain 'YearsExperience' and 'Salary' columns"}), 400
            X = data[['YearsExperience']].values
            y = data['Salary'].values
            return jsonify({"message": "Dataset loaded successfully"})
        except Exception as e:
            return jsonify({"error": f"Error loading dataset: {str(e)}"}), 400
    return jsonify({"error": "No file uploaded"}), 400

@app.route('/initialize', methods=['POST'])
def initialize():
    global learning_rate, max_epochs, costs, current_epoch, weights, bias, X, y
    params = request.json
    
    learning_rate = float(params.get('learning_rate', 0.01))
    max_epochs = int(params.get('max_epochs', 100))
    
    # Reset model state
    costs = []
    current_epoch = 0
    weights = None
    bias = None
    
    if X is None or y is None:
        return jsonify({"error": "Please load dataset first"}), 400
    
    return jsonify({"message": "Model initialized successfully"})

@app.route('/train_step', methods=['POST'])
def train_step():
    if data is None:
        return jsonify({"error": "Please upload a dataset first"}), 400
    
    global current_epoch, max_epochs
    
    if X is None or y is None:
        return jsonify({"error": "Please initialize model first"}), 400
    
    if current_epoch >= max_epochs:
        return jsonify({"message": "Training completed", "epoch": current_epoch, "cost": costs[-1] if costs else None})
    
    y_pred, cost, error = train_model_step()
    
    if error:
        return jsonify({"error": error}), 400
    
    return jsonify({
        "epoch": current_epoch,
        "cost": float(cost),
        "predictions": y_pred.tolist()
    })
def train_model_step():
    global weights, bias, costs, current_epoch, X, y
    
    if X is None or y is None:
        return None, None, "Data not initialized"
    
    if weights is None:
        weights = np.zeros(X.shape[1])
        bias = 0
    
    # Compute predictions
    y_pred = np.dot(X, weights) + bias
    
    # Compute gradients
    dw = (1/len(X)) * np.dot(X.T, (y_pred - y))
    db = (1/len(X)) * np.sum(y_pred - y)
    
    # Update parameters
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    
    # Calculate cost
    cost = np.mean((y_pred - y) ** 2)
    costs.append(cost)
    current_epoch += 1
    
    return y_pred, cost, None

@app.route('/train_all', methods=['POST'])
def train_all():
    if data is None:
        return jsonify({"error": "Please upload a dataset first"}), 400
        
    global current_epoch, max_epochs
    
    if X is None or y is None:
        return jsonify({"error": "Please initialize model first"}), 400
    
    try:
        final_predictions = None
        final_cost = None
        
        while current_epoch < max_epochs:
            y_pred, cost, error = train_model_step()
            if error:
                return jsonify({"error": error}), 400
            final_predictions = y_pred
            final_cost = cost
        
        return jsonify({
            "message": "Training completed",
            "final_cost": float(final_cost) if final_cost is not None else None,
            "predictions": final_predictions.tolist() if final_predictions is not None else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualize')
def visualize():
    if len(costs) == 0:
        return jsonify({"error": "No training data available"}), 400

    print("Costs array:", costs)  # Debug print
    print("X shape:", X.shape if X is not None else None)  # Debug print
    print("y shape:", y.shape if y is not None else None)  # Debug print

    # Create subplots
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Model Predictions', 'Training Cost'),
                       vertical_spacing=0.15)

    # Add scatter plot of actual data
    fig.add_trace(
        go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual Data',
                  marker=dict(color='blue')),
        row=1, col=1
    )

    # Add line plot of predictions if available
    if weights is not None:
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = np.dot(X_line, weights.reshape(-1, 1)) + bias
        fig.add_trace(
            go.Scatter(x=X_line.flatten(), y=y_pred.flatten(), mode='lines', 
                      name='Predictions', line=dict(color='red')),
            row=1, col=1
        )

    # Add cost history plot
    fig.add_trace(
        go.Scatter(y=costs, mode='lines+markers', name='Cost',
                  line=dict(color='green')),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text='Years of Experience', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_yaxes(title_text='Salary', row=1, col=1)
    fig.update_yaxes(title_text='Cost', row=2, col=1)

    plot_json = fig.to_json()
    print("Plot JSON:", plot_json[:200])  # Debug print first 200 chars
    return jsonify({"plot_data": json.loads(plot_json)})

@app.route('/visualize_dataset')
def visualize_dataset():
    if data is None:
        return jsonify({"error": "Please upload a dataset first"}), 400
    
    # Create scatter plot of the dataset
    fig = px.scatter(data, x='YearsExperience', y='Salary',
                    title='Dataset Visualization',
                    labels={'YearsExperience': 'Years of Experience',
                           'Salary': 'Salary'})
    
    return jsonify({"plot_data": json.loads(fig.to_json())})

@app.route('/view_data')
def view_data():
    if data is None:
        return jsonify({"error": "Please upload a dataset first"}), 400
        
    # Filter only required columns
    filtered_data = data[['YearsExperience', 'Salary']].copy()
    
    # Format YearsExperience to 2 decimal places
    filtered_data['YearsExperience'] = filtered_data['YearsExperience'].round(2)
    
    # Convert DataFrame to dictionary format suitable for display
    data_dict = {
        "columns": filtered_data.columns.tolist(),
        "data": filtered_data.values.tolist(),
        "shape": filtered_data.shape
    }
    
    return jsonify(data_dict)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    data, X, y = load_data(file)
    return jsonify({'message': 'File successfully uploaded and data processed'}), 200

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    X = np.array(data['X']).reshape(-1, 1)
    y = np.array(data['y'])
    regularization_type = data['regularization_type']
    alpha = data['alpha']
    y_test, y_pred = train_model(X, y, regularization_type, alpha)
    return jsonify({'y_test': y_test.tolist(), 'y_pred': y_pred.tolist()})

@app.route('/calculate_cost', methods=['POST'])
def calculate_cost_endpoint():
    data = request.json
    y_test = np.array(data['y_test'])
    y_pred = np.array(data['y_pred'])
    cost_function = data['cost_function']
    cost = calculate_cost(y_test, y_pred, cost_function)
    return jsonify({'cost': cost})

@app.route('/plot', methods=['POST'])
def plot():
    data = request.json
    X = np.array(data['X']).reshape(-1, 1)
    y = np.array(data['y'])
    learning_rate = data['learning_rate']
    epochs = data['epochs']
    cost_function = data['cost_function']
    fig = plot_training_steps(X, y, learning_rate, epochs, cost_function)
    return jsonify({'plot': fig.to_json()})

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    X = data['YearsExperience'].values.reshape(-1, 1)
    y = data['Salary'].values
    return data, X, y

def train_model(X, y, regularization_type, alpha):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if regularization_type == "Lasso":
        model = Lasso(alpha=alpha)
    elif regularization_type == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred

def calculate_cost(y_test, y_pred, cost_function):
    if cost_function == "Błąd średniokwadratowy (MSE)":
        cost = mean_squared_error(y_test, y_pred)
    elif cost_function == "Błąd średniobezwzględny (MAE)":
        cost = mean_absolute_error(y_test, y_pred)
    else:
        cost = r2_score(y_test, y_pred)
    return cost

def plot_training_steps(X, y, learning_rate, epochs, cost_function):
    theta = np.zeros(2)
    X_train_bias = np.c_[np.ones(X.shape[0]), X]
    cost_history = []

    for epoch in range(epochs):
        predictions = X_train_bias.dot(theta)
        errors = predictions - y
        gradient = X_train_bias.T.dot(errors) / len(y)
        theta -= learning_rate * gradient

        if cost_function == "Błąd średniokwadratowy (MSE)":
            cost = mean_squared_error(y, predictions)
        elif cost_function == "Błąd średniobezwzględny (MAE)":
            cost = mean_absolute_error(y, predictions)
        else:
            cost = r2_score(y, predictions)

        cost_history.append(cost)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(epochs)), y=cost_history, mode='lines', name='Cost'))
    return fig

if __name__ == '__main__':
    app.run(debug=True)
