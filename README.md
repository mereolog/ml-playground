# ML Playground

Welcome to the ML Playground project!

This platform provides an educational tool for understanding machine learning concepts through an interactive web interface, with a focus on code quality through comprehensive testing and a modular, extensible architecture.

## Prerequisites

- Docker

## Getting Started

1. Clone the repository to your local machine.
2. Navigate to the project root directory.

## Running the Project

To run the project, execute the following command in the project root directory:

```sh
docker compose up --build
```

## Accessing the Web Application

Once the project is running, you can access the web application at:

[http://localhost:8050/regression](http://localhost:8050/regression)

# Project Structure and Overview

## Project Overview

This project implements an interactive machine learning platform that allows users to experiment with various algorithms through a web interface. The architecture follows a microservice approach with **separate components for the web interface, machine learning API, and asset processing**. The platform is containerized using Docker for easy deployment and development.

## System Architecture

The platform consists of **three main components**:

1. **Django Web Application (`app/`)** - The frontend interface that users interact with
2. **FastAPI Machine Learning Service (`api/`)** - Backend service that implements ML algorithms
3. **Deno CSS Processor (`deno/`)** - Tailwind CSS processing service

These components communicate with each other to provide a seamless experience for experimenting with machine learning algorithms, visualizing results, and exploring datasets.

## Component Breakdown

### 1. API Service (`api/`)

The API service is built using **FastAPI** and implements machine learning algorithms that can be accessed through API endpoints and WebSockets.

#### Key Directories and Files

- **`algorithms/`**: Contains all ML algorithm implementations

  - **`base/`**: Base classes and interfaces for algorithms
    - `algorithm.py`: Abstract base class for all algorithms
    - `supervised.py`: Base class for supervised learning algorithms
    - `unsupervised.py`: Base class for unsupervised learning algorithms
  - **`supervised/`**: Implementations of supervised learning algorithms
    - `linear_regression.py`: Linear regression algorithm implementation
  - **`unsupervised/`**: Implementations of unsupervised learning algorithms

- **`schemas/configs/`**: Parameter configurations for algorithms

  - `algorithms_configs.py`: Dataclasses defining parameters for each algorithm

- **`tests/`**: Comprehensive test suite for algorithms

  - `conftest.py`: Shared test fixtures (e.g., synthetic datasets)
  - `test_base/`: Tests for base classes
  - `test_configs/`: Tests for parameter configurations
  - `test_supervised/`: Tests for supervised learning algorithms

- **`main.py`**: FastAPI application entry point, defines endpoints and WebSocket handlers

### 2. Django Web Application (`app/`)

The web application provides the **user interface** for interacting with ML algorithms, uploading datasets, and visualizing results.

#### Key Directories and Files

- **`core/`**: Django project settings and configuration

  - `settings.py`: Main Django configuration
  - `urls.py`: Root URL routing

- **`endpoints/`** and **`playgrounds/`**: Django apps for different parts of the platform

  - `views.py`: Request handlers
  - `urls.py`: URL routing for each app

- **`static/`**: Static assets including JavaScript, CSS, and images

  - **`js/src/plots/`**: Visualization code for different algorithms
    - `linear_regression.js`: Linear regression visualization
    - `linear_function.js`: Linear function visualization
  - **`js/src/utils/`**: Utility functions
    - `file-handler.js`: CSV file parsing and dataset handling
  - `websocket.js`: WebSocket client for real-time communication with API

- **`templates/`**: HTML templates
  - `base.html`: Base template with common layout
  - **`partials/`**: Reusable UI components
    - `navbar.html`: Navigation bar
    - `sidebar.html`: Sidebar with algorithm controls

### 3. Deno Service (`deno/`)

A **Tailwind CSS processor** that compiles CSS for the web application.

#### Key Files

- `src/styles.css`: Source CSS file with Tailwind directives
- `deno.json`: Deno configuration

### 4. Other Important Files

- **`docker-compose.yml`**: Defines and configures all services
- **`datasets/`**: Sample datasets for experimentation
- **`docs/`**: Documentation including test-driven development guides

## Workflow Overview

### 1. User Interaction Flow

1. User accesses the Django web application
2. User selects or uploads a dataset
3. User configures an algorithm (e.g., Linear Regression)
4. The web app establishes a WebSocket connection to the API service
5. The API processes the data and runs the selected algorithm
6. Real-time results are sent back via WebSocket
7. Results are visualized in the browser using Plotly.js

### 2. Development Workflow

The project follows a **test-driven development** approach for implementing ML algorithms:

1. Write tests defining expected algorithm behavior
2. Implement the algorithm to pass the tests
3. Refactor while ensuring tests continue to pass

## Key Features

- **Interactive ML Experimentation**: Users can experiment with different algorithms and parameters
- **Real-time Visualization**: Results update in real-time during training
- **Dataset Management**: Upload custom datasets or use provided samples
- **Extensible Architecture**: New algorithms can be added by implementing the base interfaces
- **Containerized Deployment**: All components run in Docker containers for consistent deployment

## Technology Stack

- **Frontend**: Django, Tailwind CSS, Plotly.js
- **Backend**: FastAPI, NumPy for ML algorithms
- **Infrastructure**: Docker, WebSockets
- **Development**: Test-driven development (pytest)

## License

This project is licensed under the MIT License.

