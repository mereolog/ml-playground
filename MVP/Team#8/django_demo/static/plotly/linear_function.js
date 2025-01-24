function calculateScatterPlot(lowerBound, upperBound, a, b) {
    const xValues = [];
    for (let x = lowerBound; x <= upperBound; x += 0.5) {
        xValues.push(x);
    }
    const yValues = xValues.map(x => a * x + b);
    return {
        x: xValues,
        y: yValues,
    };
}

window.onload = function() {
    const aInput = document.getElementById('aInput');
    const bInput = document.getElementById('bInput');
    const plotDOM = document.getElementById('plot');

    const aPreview = document.getElementById('aValue');
    const bPreview = document.getElementById('bValue');

    const lowerBound = -10;
    const upperBound = 10;

    const initialTrace = {
        mode: 'lines',
        type: 'scatter',
        name: 'y = mx + b',
        line: { color: 'blue', width: 2 }
    };



    function updatePlot() {

        const restyleLayout = plotDOM.layout;
        const a = parseFloat(aInput.value);
        const b = parseFloat(bInput.value);

        aPreview.textContent = a;
        bPreview.textContent = b;

        console.log(a, b);
        const traceData = calculateScatterPlot(lowerBound, upperBound, a, b);
        const trace = Object.assign({}, initialTrace, traceData);

        Plotly.react(plotDOM, [trace], restyleLayout);
    }

    document.querySelectorAll('.plotly-input').forEach(input => {
        input.addEventListener('input', updatePlot);
    });

    const layout = {
        title: 'Linear Function: y = mx + b',
        xaxis: { title: 'x', zeroline: true, range: [lowerBound, upperBound] },
        yaxis: { title: 'y', zeroline: true, range: [lowerBound, upperBound] },
        margin: { t: 40, b: 40, l: 50, r: 20 }
    };

    Plotly.newPlot(plotDOM, [], layout); // Initial plot setup

    updatePlot();
};