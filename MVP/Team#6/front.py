# pylint: disable=missing-module-docstring,missing-function-docstring
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTabWidget,
                             QComboBox, QLineEdit, QFileDialog, QMessageBox, QSlider, QHBoxLayout)
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import backend


@dataclass
class Dataset:
    """d"""
    dataset: Optional[np.ndarray] = None
    feature_data: Optional[np.ndarray] = None
    target_data: Optional[np.ndarray] = None

@dataclass
class TrainingHistory:
    """a"""
    cost_history: List[float] = field(default_factory=list)
    predictions_history: List[Tuple[float, float]] = field(default_factory=list)
    r_squared_history: List[float] = field(default_factory=list)

@dataclass
class UIComponents:
    """a"""
    slider: Optional[QSlider] = None
    controls: Optional[QWidget] = None
    next_button: Optional[QPushButton] = None
    previous_button: Optional[QPushButton] = None

class LinearRegressionApp(QWidget):
    """a"""
    def __init__(self):
        super().__init__()
        self.init_ui()

        self.dataset = Dataset()
        self.config = backend.Config
        self.training_history = TrainingHistory()
        self.ui_components = UIComponents()

        self.resize(700, 1000)  # Width x Height

    def init_ui(self):
        self.layout = QVBoxLayout()

        # Load Dataset
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.layout.addWidget(self.load_button)

        # Cost Function Dropdown
        self.cost_label = QLabel("Select Cost Function:")
        self.cost_dropdown = QComboBox()
        self.cost_dropdown.addItems(["MSE", "MAE", "R-squared"])

        # Disable the "MAE" option
        item = self.cost_dropdown.model().item(2)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)

        self.cost_dropdown.currentIndexChanged.connect(self.select_cost_function)
        self.layout.addWidget(self.cost_label)
        self.layout.addWidget(self.cost_dropdown)

        # Learning Rate Input
        self.lr_label = QLabel("Learning Rate:")
        self.lr_input = QLineEdit("0.01")
        self.lr_input.textChanged.connect(self.set_learning_rate)
        self.layout.addWidget(self.lr_label)
        self.layout.addWidget(self.lr_input)

        # How long will it take ??
        self.iter_label = QLabel("Number of iterations:")
        self.iterations_input = QLineEdit("1000")
        self.iterations_input.textChanged.connect(self.set_iters)
        self.layout.addWidget(self.iter_label)
        self.layout.addWidget(self.iterations_input)

        self.reg_label = QLabel("Regularization:")
        self.reg_dropdown = QComboBox()
        self.reg_dropdown.addItems(["None", "Lasso", "Ridge"])
        self.reg_dropdown.currentIndexChanged.connect(self.select_regularization)
        self.layout.addWidget(self.reg_label)
        self.layout.addWidget(self.reg_dropdown)

        self.lambda_label = QLabel("Regularization Parameter:")
        self.lambda_input = QLineEdit("0.1")
        self.lambda_input.textChanged.connect(self.set_lambda)
        self.layout.addWidget(self.lambda_label)
        self.layout.addWidget(self.lambda_input)

        # Execution Buttons
        self.run_all_button = QPushButton("Run All Steps")
        self.run_all_button.clicked.connect(self.run_all_steps)
        self.layout.addWidget(self.run_all_button)

        # Plot Widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Create the first tab for the original plot
        self.first_tab = QWidget()
        self.first_tab_layout = QVBoxLayout()
        self.first_tab.setLayout(self.first_tab_layout)

        # Add a plot to the first tab
        self.plot = PlotCanvas(self)
        self.first_tab_layout.addWidget(self.plot)

        # Add the first tab to the tab widget
        self.tabs.addTab(self.first_tab, "Regression Plot")

        # Create the second tab for the additional plot
        self.second_tab = QWidget()
        self.second_tab_layout = QVBoxLayout()
        self.second_tab.setLayout(self.second_tab_layout)

        # Add a plot to the second tab
        self.second_tab_plot = PlotCanvas(self)
        self.second_tab_layout.addWidget(self.second_tab_plot)

        # Add the second tab to the tab widget
        self.tabs.addTab(self.second_tab, "Cost Plot")

        # Results Section
        self.cost_output = QLabel("Cost: N/A")
        self.layout.addWidget(self.cost_output)
        self.params_output = QLabel("Model Parameters: N/A")
        self.layout.addWidget(self.params_output)
        self.r_squared = QLabel("R-squared: N/A")
        self.layout.addWidget(self.r_squared)

        self.setLayout(self.layout)
        self.setWindowTitle("Linear Regression")

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV Files (*.csv)")
        if file_path:
            try:
                (self.dataset.feature_data,
                 self.dataset.target_data) = backend.load_dataset(file_path)
                (self.dataset.feature_data,
                 self.dataset.target_data) = backend.standarization(self.dataset.feature_data,
                                                                        self.dataset.target_data)
                self.plot.plot_static(self.dataset.feature_data, self.dataset.target_data)
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except FileNotFoundError:
                QMessageBox.critical(self, "Error", "The specified file was not found.")
            except pd.errors.EmptyDataError:
                QMessageBox.critical(self, "Error", "The CSV file is empty.")
            except ValueError as e:
                QMessageBox.critical(self, "Error", f"Invalid data in the CSV file: {e}")

    def select_cost_function(self, index):
        if index == 0:
            self.config.cost_function = backend.mean_squared_error
        elif index == 1:
            self.config.cost_function = backend.mean_absolute_error
        elif index == 2:
            self.config.cost_function = backend.r_squared

    def set_learning_rate(self, text):
        try:
            self.config.learning_rate = float(text)
        except ValueError:
            pass

    def set_iters(self, number):
        self.config.num_iterations = int(number)

    def select_regularization(self, index):
        if index == 0:
            self.config.regularization = None
        elif index == 1:
            self.config.regularization = "L1"
        elif index == 2:
            self.config.regularization = "L2"

    def set_lambda(self, text):
        try:
            self.config.regularization_coefficient = float(text)
        except ValueError:
            pass

    def run_all_steps(self):
        if self.dataset.feature_data is None or self.dataset.target_data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        try:
            config = backend.Config(learning_rate=self.learning_rate,
                                    num_iterations=self.num_iterations,
                                    error_func=self.cost_function,
                                    regularization=self.regularization,
                                    regularization_coefficient=self.regularization_coefficient)
            (_, _, self.training_history.predictions_history,
             self.training_history.cost_history,
             self.training_history.r_squared_history) = backend.gradient_descent(
                self.dataset.feature_data, self.dataset.target_data, config
            )

            self.plot_cost_curve()
            self.add_time_machine()
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input values: {e}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Runtime error during gradient descent: {e}")

    def add_time_machine(self):

        if self.ui_components.slider is None:
            self.ui_components.slider = QSlider(QtCore.Qt.Horizontal)
            self.ui_components.slider.valueChanged.connect(self.see_past)
            self.layout.addWidget(self.ui_components.slider)
            self.ui_components.slider.setMinimum(0)
            self.ui_components.slider.setMaximum(len(self.training_history.predictions_history) - 1)
            self.ui_components.slider.setValue(0)

        if self.ui_components.controls is None:
            self.button_layout = QHBoxLayout()

            self.ui_components.previous_button = QPushButton("Previous")
            self.button_layout.addWidget(self.ui_components.previous_button)
            self.ui_components.previous_button.clicked.connect(lambda: self.move_slider(-1))

            self.ui_components.next_button = QPushButton("Next")
            self.button_layout.addWidget(self.ui_components.next_button)
            self.ui_components.next_button.clicked.connect(lambda: self.move_slider(1))

            self.layout.addLayout(self.button_layout)
            self.setLayout(self.layout)

            self.ui_components.controls = "na pewno nie None"

    def move_slider(self, step):
        # Move the slider by the specified step
        current_value = self.ui_components.slider.value()
        new_value = current_value + step
        min_value = self.ui_components.slider.minimum()
        max_value = self.ui_components.slider.maximum()

        if min_value <= new_value <= max_value:
            self.ui_components.slider.setValue(new_value)

    def see_past(self):
        # Get the current slider value that dictates which prediction to draw
        index = self.ui_components.slider.value()
        m, b = self.training_history.predictions_history[index]
        predictions = m * self.dataset.feature_data + b

        # Just update the line on the plot
        self.plot.update_dynamic_line(self.dataset.feature_data, predictions)
        self.params_output.setText(f"Model Parameters: m = {m:.4f},"
                                   f" b = {b:.4f}, iteration = {index}")
        self.cost_output.setText(f"Cost: {self.training_history.cost_history[index]:.4f}")
        self.r_squared.setText(f"R-squared cost: "
                               f"{self.training_history.r_squared_history[index]:.4f}")

    def plot_cost_curve(self):
        iterations = np.arange(len(self.training_history.cost_history))
        self.second_tab_plot.plot_cost_curve(iterations, self.training_history.cost_history)


class PlotCanvas(FigureCanvas):
    """d"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.line = None  # Placeholder for the dynamic line

    def plot_static(self, x, y):
        """Plot static elements """
        self.ax.scatter(x, y, color='blue', label='Actual Data')
        self.ax.set_xlabel("Years of Experience")
        self.ax.set_ylabel("Salary")
        self.ax.legend()
        self.draw()

    def update_dynamic_line(self, x, predictions):
        """Update the line plot for dynamic elements."""
        if self.line is not None:
            # Remove the previous line if it exists
            self.line.remove()

        # Plot the new line
        self.line, = self.ax.plot(x, predictions, color='red', label='Predicted Data')

        # Update the canvas
        self.draw()

    def plot_cost_curve(self, iterations, cost_history):
        """Plot the cost curve."""
        self.ax.clear()
        self.ax.plot(iterations, cost_history, color='red', label='Cost over Iterations')
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Cost")
        self.ax.legend()
        self.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinearRegressionApp()
    window.show()
    sys.exit(app.exec_())
