const address = 'ws://' + window.location.hostname + ':8000/ws';
console.log('address', address);
const socket = new WebSocket(address);




function toggleDrawer() {
  const drawer = document.querySelector('input[type="checkbox"].drawer-toggle');
    if (drawer) {
        drawer.checked = false; // This closes the drawer
    }
}


function updatePlot(plotDOM, initialTrace, traceData) {

  const restyleLayout = plotDOM.layout;
  const trace = Object.assign({}, initialTrace, traceData);

  Plotly.react(plotDOM, [trace], restyleLayout);
}

function updateMetricPlot(plotDOM, data) {
  const r2MetricDOM = document.getElementById('r2-metric');
  const mseMetricDOM = document.getElementById('mse-metric');
  const rmseMetricDOM = document.getElementById('rmse-metric');
  const maeMetricDOM =  document.getElementById('mae-metric');

  r2MetricDOM.innerHTML = data.r2_values[data.r2_values.length - 1].toFixed(4);
  mseMetricDOM.innerHTML = data.mse_values[data.mse_values.length - 1].toFixed(4);
  rmseMetricDOM.innerHTML = data.rmse_values[data.rmse_values.length - 1].toFixed(4);
  maeMetricDOM.innerHTML = data.mae_values[data.mae_values.length - 1].toFixed(4);



  var layout = {
    grid: {rows: 2, columns: 2, pattern: 'independent'},
    showlegend: false,
    margin: { l: 60, r: 60, b: 30, t: 30, pad: 4 },
    xaxis: {title: 'epochs'},
    yaxis: {title: 'R2'},
    xaxis2: {title: 'epochs'},
    yaxis2: {title: 'MSE'},
    xaxis3: {title: 'epochs'},
    yaxis3: {title: 'RMSE'},
    xaxis4: {title: 'epochs'},
    yaxis4: {title: 'MAE'}
    };

  var trace1 = {
    name: 'R2',
    mode: 'lines',
    type: 'scatter',
    x: data.epochs,
    y: data.r2_values,
    xaxis: 'x1',
    yaxis: 'y1',
  };

  var trace2 = {
    name: 'MSE',
    mode: 'lines',
    type: 'scatter',
    x: data.epochs,
    y: data.mse_values,
    xaxis: 'x2',
    yaxis: 'y2',
  };

  var trace3 = {
    name: 'RMSE',
    mode: 'lines',
    type: 'scatter',
    x: data.epochs,
    y: data.rmse_values,
    xaxis: 'x3',
    yaxis: 'y3',
  };

  var trace4 = {
    name: 'MAE',
    mode: 'lines',
    type: 'scatter',
    x: data.epochs,
    y: data.mae_values,
    xaxis: 'x4',
    yaxis: 'y4',
  };

  var tracesData = [trace1, trace2, trace3, trace4];

  var options = { displayModeBar: false };

  Plotly.react(plotDOM, tracesData, layout, options);
}

function updateLossPlot(plotDOM, data) {

  layout = plotDOM.layout;
  
  lossTrace = {
    name: 'loss',
    mode: 'lines',
    type: 'scatter',
    x: data.epochs,
    y: data.loss_values,
  }

  console.log(lossTrace);

  var options = { displayModeBar: false };

  Plotly.react(plotDOM, [lossTrace], layout, options);
}


function calculateTrainingProgress(current_epoch_data) {
  const epochs_num = document.getElementById("epochs").value;
  current_epoch = current_epoch_data[current_epoch_data.length - 1];
  console.log('calc-data', current_epoch, epochs_num);
  return Math.round(current_epoch / epochs_num * 100) + 1;
}

document.addEventListener('DOMContentLoaded', function() {


socket.onopen = function(e) {
  console.log("[open] Connection established");
}

socket.onclose = function(e) {
  console.log("[close] Connection closed");
};

socket.onerror = function(e) {
  console.error("[error] Connection error", e);
};

socket.onmessage = function(event) {
  const data = JSON.parse(event.data);

  const regressionPlotDOM = document.getElementById('regression-plot');
  const metricsPlotDOM = document.getElementById('metrics-plot');
  const lossPlotDOM = document.getElementById('loss-plot');
  const progressBarDOM = document.getElementById('training-progress');

  const datasetTrace = {
    name: 'dataset scatter',
    mode: 'markers',
    type: 'scatter',
    x: data.dependent_variable,
    y: data.independent_variable,
  };

  const predictionTrace = {
    name: 'prediction line',
    mode: 'lines',
    type: 'scatter',
    x: data.x_plot,
    y: data.y_pred,
  }

  const trace = [predictionTrace, datasetTrace];

  const restyleLayout = regressionPlotDOM.layout;

  if (data.task === 'update_plot') {
    console.log("Updating plot...");

    const progress = calculateTrainingProgress(data.epochs, data.current_epoch);
    // modify value attribute of the progress bar
    console.log('progress', progress);
    progressBarDOM.value = progress


    Plotly.react(regressionPlotDOM, trace, restyleLayout);

    updateMetricPlot(metricsPlotDOM, data);
    updateLossPlot(lossPlotDOM, data);

  }
  if (data.task === 'initialize_plots') {
    console.log("Initializing plot...");
    Plotly.react(regressionPlotDOM, trace, restyleLayout);
  }

  // updatePlot(plotDOM, initialTrace, traceData);

}


});


function prepareModel(event) {
  toggleDrawer();
  console.log("Preparing model...");

  dependentVariable = document.getElementById("dependent-variable").value;
  independentVariable = document.getElementById("independent-variable").value;
  learningRate = document.getElementById("learning-rate").value;
  epochs = document.getElementById("epochs").value;
  regularization = document.getElementById("regularization").value;

  parameters = {
    "task": "prepare_model",
    "dependent_variable": dependentVariable,
    "independent_variable": independentVariable,
    "learning_rate": learningRate,
    "epochs": epochs,
    "regularization": regularization
  }

  datasetFile = window.dataset;



  const reader = new FileReader();
  reader.onload = function(event) {
    const fileContent = event.target.result;
    parameters["dataset"] = fileContent;
    socket.send(JSON.stringify(parameters));
  };
  reader.readAsText(datasetFile);

  event.preventDefault();
}

function trainModel(event) {
  toggleDrawer();
  console.log("Training model...");

  parameters = {
    "task": "train_model",
  }

  socket.send(JSON.stringify(parameters));

  event.preventDefault();
}


// var ws = new WebSocket("ws://localhost:8000/ws");
// ws.onmessage = function(event) {
//     var messages = document.getElementById('messages')
//     var message = document.createElement('li')
//     var content = document.createTextNode(event.data)
//     message.appendChild(content)
//     messages.appendChild(message)

// };
// function sendMessage(event) {
//     var input = document.getElementById("messageText")
//     ws.send(input.value)
//     input.value = ''
//     event.preventDefault()
// }