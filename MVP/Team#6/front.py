import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTabWidget,
                             QComboBox, QLineEdit, QFileDialog, QMessageBox, QSlider, QHBoxLayout)
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import backend as backend


class LinearRegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Dataset-related attributes
        self.dataset = None
        self.feature_data = None
        self.target_data = None

        # Model parameters
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.cost_function = backend.mean_squared_error
        self.regularization = None
        self.regularization_coefficient = 0.1
        self.model_params = None

        # History tracking
        self.cost_history = []
        self.predictions_history = []
        self.r_squared_history = []

        # UI-related attributes
        self.slider = None
        self.controls = None
        self.next_button = None
        self.previous_button = None

        # Set initial window size
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

        # Regularization Type
        self.reg_label = QLabel("Regularization:")
        self.reg_dropdown = QComboBox()
        self.reg_dropdown.addItems(["None", "Lasso", "Ridge"])
        self.reg_dropdown.currentIndexChanged.connect(self.select_regularization)
        self.layout.addWidget(self.reg_label)
        self.layout.addWidget(self.reg_dropdown)

        # Regularization Parameter
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
                self.feature_data, self.target_data = backend.load_dataset(file_path)
                self.feature_data, self.target_data = backend.standarization(self.feature_data, self.target_data)
                self.plot.plot_static(self.feature_data, self.target_data)  # Plot the static elements once
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def select_cost_function(self, index):
        if index == 0:
            self.cost_function = backend.mean_squared_error
        elif index == 1:
            self.cost_function = backend.mean_absolute_error
        elif index == 2:
            self.cost_function = backend.r_squared

    def set_learning_rate(self, text):
        try:
            self.learning_rate = float(text)
        except ValueError:
            pass

    def set_iters(self, number):
        self.num_iterations = int(number)

    def select_regularization(self, index):
        if index == 0:
            self.regularization = None
        elif index == 1:
            self.regularization = "L1"
        elif index == 2:
            self.regularization = "L2"

    def set_lambda(self, text):
        try:
            self.regularization_coefficient = float(text)
        except ValueError:
            pass

    def run_all_steps(self):
        if self.feature_data is None or self.target_data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        try:
            # Call the gradient descent method and gather predictions
            i, j, self.predictions_history, self.cost_history, self.r_squared_history = backend.gradient_descent(
                self.feature_data, self.target_data,
                learning_rate=self.learning_rate,
                num_iterations=self.num_iterations,
                error_func=self.cost_function,
                regularization=self.regularization,
                regularization_coefficient=self.regularization_coefficient,
            )

            self.plot_cost_curve()
            self.add_time_machine()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during all steps: {e}")

    def add_time_machine(self):

        if self.slider is None:
            self.slider = QSlider(QtCore.Qt.Horizontal)
            self.slider.valueChanged.connect(self.see_past)
            self.layout.addWidget(self.slider)
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.predictions_history) - 1)
            self.slider.setValue(0)

        if self.controls is None:
            self.button_layout = QHBoxLayout()

            self.previous_button = QPushButton("Previous")
            self.button_layout.addWidget(self.previous_button)
            self.previous_button.clicked.connect(lambda: self.move_slider(-1))

            self.next_button = QPushButton("Next")
            self.button_layout.addWidget(self.next_button)
            self.next_button.clicked.connect(lambda: self.move_slider(1))

            self.layout.addLayout(self.button_layout)
            self.setLayout(self.layout)

            self.controls = "na pewno nie None"

    def move_slider(self, step):
        # Move the slider by the specified step
        current_value = self.slider.value()
        new_value = current_value + step
        min_value = self.slider.minimum()
        max_value = self.slider.maximum()

        if min_value <= new_value <= max_value:
            self.slider.setValue(new_value)

    def see_past(self):
        # Get the current slider value that dictates which prediction to draw
        index = self.slider.value()
        m, b = self.predictions_history[index]
        predictions = m * self.feature_data + b

        # Just update the line on the plot
        self.plot.update_dynamic_line(self.feature_data, predictions)
        self.params_output.setText(f"Model Parameters: m = {m:.4f}, b = {b:.4f}, iteration = {index}")
        self.cost_output.setText(f"Cost: {self.cost_history[index]:.4f}")
        self.r_squared.setText(f"R-squared cost: {self.r_squared_history[index]:.4f}")

    def plot_cost_curve(self):
        iterations = np.arange(len(self.cost_history))
        self.second_tab_plot.plot_cost_curve(iterations, self.cost_history)


class PlotCanvas(FigureCanvas):
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
