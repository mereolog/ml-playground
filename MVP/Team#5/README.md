# Linear Regression Playground

This project is a simple web application for performing linear regression analysis using Streamlit. It allows users to upload a CSV file containing data, select different cost functions and regularization methods, and visualize the training process.

## Files

### app.py
This file is the main entry point of the application. It uses Streamlit to create a web interface where users can upload their CSV files, select cost functions, regularization types, and other hyperparameters for training a linear regression model.

### model_logic.py
This file contains the core logic for loading data, training the model, calculating cost functions, and plotting training steps. It supports different regularization techniques like Lasso and Ridge, and provides multiple cost functions such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 Score.

### requirements.txt
This file lists the dependencies required to run the application, including Streamlit, Pandas, NumPy, scikit-learn, and Plotly.

## Getting Started

To run the application, make sure you have Python installed, and then install the required packages using:

```bash
pip install -r requirements.txt
```

After installing the dependencies, you can start the application by running:

```bash
streamlit run app.py
```

## Usage

1. Upload a CSV file containing `YearsExperience` and `Salary` columns.
2. Select the desired cost function and regularization type.
3. Adjust hyperparameters such as learning rate, regularization parameter (alpha), and epochs.
4. View the results and cost function value.