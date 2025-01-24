function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            
            complete: function(results) {
                const data = results.data;
                const columns = results.meta.fields;

                const columnTypes = inferColumnTypes(data, columns);
                // If the first column is empty, it is an index column
                if (columns[0] === '' ) columns[0] = 'Index';

                resolve({
                    rows: data.length,
                    columns: columns,
                    columnTypes: columnTypes
                });
            },
            error: function(error) {
                reject(error);
            }
        });
    });
}

function inferColumnTypes(data, columns) {
    const columnTypes = {};

    columns.forEach(column => {
        const value = data.map(row => row[column])[0];
        const type = inferTypes(value);
        columnTypes[column] = type;
    });

    return columnTypes;
}

function inferTypes(value) {

        function isValidDate(value) {
            const date = new Date(value);
            return date instanceof Date && !isNaN(date);
        }

    let valueTypes = [];

    let type = typeof value;

    if(type === 'object') {
        if(value === null) {
            valueTypes.push('null');
        } else {
            valueTypes.push('object');
        }
    } else if(type === undefined) {
        valueTypes.push('undefined');
    } else if(type === 'string') {
        valueTypes.push('string');
    } else if(type === 'number'){
        valueTypes.push(value % 1 === 0 ? 'integer' : 'float');
    } else if(type === 'boolean') {
        valueTypes.push('boolean');
    } 

    return valueTypes.join(' / ');
}

function updateTable(tableID, data) {
    const tableDOM = document.getElementById(tableID);
    const tableBody = tableDOM.tBodies[0]
    tableBody.innerHTML = '';
    if (!tableDOM) return;

    console.log('Data:', data);

    const columnDataTypes = Object.values(data.columnTypes);
    const columnNames = data.columns;

    columnNames.map((column, index) => {
        const row = document.createElement('tr');

        const columnNameCell = document.createElement('td');
        columnNameCell.textContent = column;

        const dataTypeCell = document.createElement('td');
        dataTypeCell.textContent = columnDataTypes[index];

        row.appendChild(columnNameCell);
        row.appendChild(dataTypeCell);
        tableBody.appendChild(row);
    });

};

function handleCSVData(file) {
    
    if (file) {
        console.log('File selected:', file.name);
        document.getElementById('csv-filename').textContent = `${file.name}`;
        document.getElementById('csv-size').textContent = `Size: ${(file.size / 1024).toFixed(2)} KB`;

        parseCSV(file).then(info => {
            document.getElementById('csv-rows').textContent = `Rows: ${info.rows}`;
            document.getElementById('csv-columns-count').textContent = `Columns: ${info.columns.length}`;

            const dependentSelect = document.getElementById('dependent-variable');
            const independentSelect = document.getElementById('independent-variable');
            dependentSelect.innerHTML = '<option value="">Select a column</option>';
            independentSelect.innerHTML = '<option value="">Select a column</option>';

            info.columns.forEach(column => {
                const option = document.createElement('option');
                option.value = column;
                option.textContent = column;
                dependentSelect.appendChild(option);
                independentSelect.appendChild(option.cloneNode(true));
            });

            // Update the table with column names and types
            updateTable('dataset-spec-table', info);
        }).catch(error => {
            console.error('Error parsing CSV:', error);
        });
    } else {
        console.log('No file selected');
    }
}

const DatasetInputDOM = document.getElementById('dataset-input');
const ExistingDatasetSelectDOM = document.getElementById('existing-dataset-select');


document.addEventListener('DOMContentLoaded', function() {
    DatasetInputDOM.addEventListener('change', function(event) {
        ExistingDatasetSelectDOM.value = 0;
        const file = event.target.files[0];
        window.dataset = value;
        handleCSVData(file);
    });

    ExistingDatasetSelectDOM.addEventListener('change', function(event) {
        DatasetInputDOM.value = '';
        const value = event.target.value;
        
        function processDataset(dataset_file_name) {
            fetch('http://' + window.location.hostname + ':8050/api/load_file/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie("csrftoken") // Assuming you're using Django
                },
                body: JSON.stringify({
                    datasetFileName: dataset_file_name
                })
            })
            .then(response => response.blob())
            .then(blob => {
                const file = new File([blob], dataset_file_name, { type: 'text/csv' });
                window.dataset = file;
                handleCSVData(file);
            });
        }

        if (value) {
            console.log('Selected dataset:', value);
            processDataset(value);
        }
    });
});

