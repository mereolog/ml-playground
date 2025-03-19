import logging
from typing import Tuple
from io import StringIO
from torch.autograd import Variable
from models.linear_regression import LinearRegressionModel

import torch
import pandas as pd
from fastapi import FastAPI, WebSocket

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

connected_clients = []


class MetricsCalculator:
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred)).item()

    @staticmethod
    def r2_score(y_true, y_pred):
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total).item()

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2).item()

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)  # Recalculate MSE as a tensor
        return torch.sqrt(mse).item()


class DataProcessor:
    @staticmethod
    def normalize_data(data):
        return (data - data.min()) / (data.max() - data.min())

    @staticmethod
    def prepare_data(csv_content: str, dependent_variable: str, independent_variable: str) -> Tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(StringIO(csv_content))

        if df.columns[0] == "Unnamed: 0":
            df.rename(columns={"Unnamed: 0": "Index"}, inplace=True)

        dependent_data = torch.from_numpy(df[dependent_variable].values).unsqueeze(1).float()
        independent_data = torch.from_numpy(df[independent_variable].values).unsqueeze(1).float()

        return DataProcessor.normalize_data(dependent_data), DataProcessor.normalize_data(independent_data)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        logging.info("Received data: %s", data)

        if data["task"] == "prepare_model":
            csv_string_io = StringIO(data["dataset"])
            df = pd.read_csv(csv_string_io)

            if df.columns[0] == "Unnamed: 0":
                df.rename(columns={"Unnamed: 0": "Index"}, inplace=True)

            dependent_variable = data["dependent_variable"]
            independent_variable = data["independent_variable"]

            dependent_variable_data = df[dependent_variable].values
            independent_variable_data = df[independent_variable].values

            x_data_torch = Variable(torch.from_numpy(dependent_variable_data).unsqueeze(1).float())
            y_data_torch = Variable(torch.from_numpy(independent_variable_data).unsqueeze(1).float())

            x_data_norm = DataProcessor.normalize_data(x_data_torch)
            y_data_norm = DataProcessor.normalize_data(y_data_torch)

            model_parameters = {
                "epochs": int(data["epochs"]),
                "learning_rate": float(data["learning_rate"]),
            }

            model = LinearRegressionModel()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=model_parameters["learning_rate"])

            with torch.no_grad():
                y_plot = model(x_data_norm).detach().numpy().flatten()

            response_data = {
                "task": "initialize_plots",
                "dependent_variable": DataProcessor.normalize_data(torch.from_numpy(dependent_variable_data)).tolist(),
                "independent_variable": DataProcessor.normalize_data(torch.from_numpy(independent_variable_data)).tolist(),
                "x_plot": x_data_norm.flatten().tolist(),
                "y_pred": y_plot.tolist(),
            }

            await websocket.send_json(response_data)

        elif data["task"] == "train_model":
            loss_values, mae_values, mse_values, rmse_values, r2_values = [], [], [], [], []
            epochs = model_parameters["epochs"]

            for epoch in range(epochs):
                pred_y = model(x_data_norm)
                loss = criterion(pred_y, y_data_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())
                mae_values.append(MetricsCalculator.mean_absolute_error(y_data_norm, pred_y))
                mse_values.append(MetricsCalculator.mean_squared_error(y_data_norm, pred_y))
                rmse_values.append(MetricsCalculator.root_mean_squared_error(y_data_norm, pred_y))
                r2_values.append(MetricsCalculator.r2_score(y_data_norm, pred_y))

                if epoch % 10 == 0:
                    logging.info("Epoch: %d, Loss: %f", epoch, loss.item())

                    update_data = {
                        "task": "update_plot",
                        "dependent_variable": DataProcessor.normalize_data(torch.from_numpy(dependent_variable_data)).tolist(),
                        "independent_variable": DataProcessor.normalize_data(torch.from_numpy(independent_variable_data)).tolist(),
                        "x_plot": x_data_norm.flatten().tolist(),
                        "y_pred": pred_y.flatten().tolist(),
                        "loss_values": loss_values,
                        "mae_values": mae_values,
                        "mse_values": mse_values,
                        "rmse_values": rmse_values,
                        "r2_values": r2_values,
                        "epochs": list(range(epoch + 1)),
                        "current_epoch": epoch,
                    }

                    await websocket.send_json(update_data)

        logging.info("Sending data: %s", data)
        await websocket.send_json(data)
