import json
import os
from io import StringIO

import pandas as pd
from flask import Flask, render_template, request, session, send_from_directory
from pydantic import ValidationError

from algorithms.supervised.decision_trees import DecisionTree
from schemas.configs.decision_tree_config import DecisionTreeParams

SAMPLE_FILES_DIR = os.path.join(os.path.dirname(__file__), 'samples')
app = Flask(__name__)
app.secret_key = 'niewiem'

@app.before_request
def clear_session_on_refresh():
    if request.method == 'GET' and not request.args:
        session.pop('csv_data', None)
        session.pop('columns', None)

@app.route('/samples/<filename>')
def download_sample(filename):
    return send_from_directory(SAMPLE_FILES_DIR, filename)

def get_sample_files():
    """Helper function to get list of available sample files"""
    try:
        files = [f for f in os.listdir(SAMPLE_FILES_DIR) if f.endswith('.csv')]
        return files
    except FileNotFoundError:
        return []

def run_decision_tree_algorithm(csv_data, features, target, params):
    try:
        print(params)
        data = pd.read_csv(StringIO(csv_data))
        tree = DecisionTree(params=params)
        tree.fit(data, features, target)
        json_tree = json.loads(tree.to_json())
        return json_tree.get("tree", {})
    except Exception as e:
        return {'error': str(e)}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'sample_file' in request.form:
            return handle_sample_file_upload()
        elif 'csv_data' in request.form:
            return handle_algorithm_run()

    return render_index()

def handle_sample_file_upload():
    sample_filename = request.form['sample_file']
    try:
        sample_path = os.path.join(SAMPLE_FILES_DIR, sample_filename)
        with open(sample_path, 'r', encoding='utf-8') as f:
            csv_data = f.read()
        data = pd.read_csv(StringIO(csv_data))
        csv_head = data.to_html(classes='table table-striped')
        session['csv_data'] = csv_data
        session['columns'] = data.columns.tolist()
        session['csv_head'] = csv_head
        return render_index(
            columns=data.columns.tolist(),
            selected_sample=sample_filename,
            csv_head=csv_head
        )
    except Exception as e:
        return render_index(error=str(e))

def handle_algorithm_run():
    try:
        csv_data = session.get('csv_data')
        features = request.form.getlist('features')
        target = request.form['target']
        csv_head = session.get('csv_head')


        params_input = {
            "max_tree_depth": int(request.form.get('max_depth', 4)),
            "min_split_samples": int(request.form.get('min_samples_split', 2))
        }
        print(params_input)

        try:
            params = DecisionTreeParams(**params_input)
        except ValidationError as e:
            raise ValueError(f"Invalid parameters: {e}")

        if target in features:
            raise ValueError("Target column must be different from features.")

        result = run_decision_tree_algorithm(csv_data, features, target, params)
        return render_index(
            result=result,
            selected_features=features,
            selected_target=target,
            max_tree_depth=params.max_tree_depth,
            min_split_samples=params.min_split_samples,
            csv_head=csv_head
        )
    except Exception as e:
        return render_index(
            error=str(e),
            selected_features=request.form.getlist('features'),
            selected_target=request.form['target'],
            max_tree_depth=request.form.get('max_depth', 4),
            min_split_samples=request.form.get('min_samples_split', 2),
            csv_head=session.get('csv_head')
        )

def render_index(**kwargs):
    defaults = {
        'columns': session.get('columns', []),
        'sample_files': get_sample_files(),
        'error': None,
        'max_tree_depth': kwargs.get('max_tree_depth', 4),
        'min_split_samples': kwargs.get('min_split_samples', 2)
    }
    print(defaults)
    return render_template('index.html', **{**defaults, **kwargs})

if __name__ == '__main__':
    app.run(debug=True)