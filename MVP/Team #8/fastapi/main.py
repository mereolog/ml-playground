from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
app = FastAPI()

import logging
import pandas as pd
from io import StringIO

import torch
from torch.autograd import Variable
from models.linear_regression import LinearRegressionModel

# Configure logging
logging.basicConfig(level=logging.INFO)

connected_clients = []

      
def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def r2_score(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total).item()

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()

def root_mean_squared_error(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)  # Recalculate MSE as a tensor
    return torch.sqrt(mse).item()
    
def normalize_data(data):
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data
    
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        logging.info(f"Received data: {data}")
        
        if data['task'] == 'prepare_model':
            csvStringIO = StringIO(data['dataset'])
            df = pd.read_csv(csvStringIO, sep=",")
        
            if df.columns[0] == 'Unnamed: 0':
                df.rename( columns={'Unnamed: 0':'Index'}, inplace=True )
            
            dependent_variable = data['dependent_variable']
            independent_variable = data['independent_variable']
            
            dependent_variable_data = df[dependent_variable].values
            independent_variable_data = df[independent_variable].values
            
            
        
        
            model_parameters = {
                "epochs": int(data['epochs']),
                "learning_rate": float(data['learning_rate']),
                  
            }
            
            model = LinearRegressionModel()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=model_parameters['learning_rate'])
            
            x_data_torch = Variable(torch.from_numpy(dependent_variable_data).unsqueeze(1).float())
            y_data_torch = Variable(torch.from_numpy(independent_variable_data).unsqueeze(1).float())
            x_data_norm = normalize_data(x_data_torch)
            y_data_norm = normalize_data(y_data_torch)
            
            with torch.no_grad():
                y_plot = model(x_data_norm).detach().numpy().flatten()
                        
            data = {
                "task": "initialize_plots",
                "dependent_variable": normalize_data(dependent_variable_data).tolist(),
                "independent_variable": normalize_data(independent_variable_data).tolist(),
                "x_plot": x_data_norm.flatten().tolist(),
                "y_pred": y_plot.tolist(),
            }
            
            
            await websocket.send_json(data)
            
            
        elif data['task'] == 'train_model':
            loss_values = []
            
            mae_values = []
            mse_values = []
            rmse_values = []
            r2_values = []
            
            epochs = model_parameters['epochs']
            
            for epoch in range(epochs):
                pred_y = model(x_data_norm)
                loss = criterion(pred_y, y_data_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
                
                mae_values.append(mean_absolute_error(y_data_norm, pred_y))
                mse_values.append(mean_squared_error(y_data_norm, pred_y))
                rmse_values.append(root_mean_squared_error(y_data_norm, pred_y))
                r2_values.append(r2_score(y_data_norm, pred_y))
                
                if epoch % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Loss: {loss.item()}")
                    data = {
                    "task": "update_plot",
                    "dependent_variable": normalize_data(dependent_variable_data).tolist(),
                    "independent_variable": normalize_data(independent_variable_data).tolist(),
                    "x_plot": x_data_norm.flatten().tolist(),
                    "y_pred": pred_y.flatten().tolist(),
                    "loss_values": loss_values,
                    "mae_values": mae_values,
                    "mse_values": mse_values,
                    "rmse_values": rmse_values,
                    "r2_values": r2_values,
                    "epochs": [i for i in range(epoch+1)],
                    "current_epoch": epoch,
                    }
                    
                    await websocket.send_json(data)
                
        
        
        logging.info(f"Sending data: {data}")
        await websocket.send_json(data)
        
        
        
        