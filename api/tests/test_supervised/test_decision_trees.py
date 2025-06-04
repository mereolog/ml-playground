import pandas as pd
import pytest
from algorithms.supervised.decision_trees import DecisionTree, TreeNode


@pytest.fixture
def sample_data():
    data = {
        'Outlook': ['Sunny', 'Overcast', 'Rain', 'Sunny', 'Sunny', 'Overcast'],
        'Temperature': ['Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool'],
        'Humidity': ['High', 'High', 'High', 'Normal', 'Normal', 'Normal'],
        'Windy': ['False', 'True', 'False', 'False', 'True', 'True'],
        'PlayTennis': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)


def test_sanity():
    tree = DecisionTree(max_depth=5, min_samples_split=2)
    assert isinstance(tree, DecisionTree)

def test_initialization():
    tree = DecisionTree(max_depth=5, min_samples_split=2)
    assert tree.max_depth == 5
    assert tree.min_samples_split == 2

def test_tree_fit(sample_data):
    features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    label = 'PlayTennis'
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(sample_data, features, label)
    assert tree.root is not None
    assert isinstance(tree.root, TreeNode)

def test_predict(sample_data):
    features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    label = 'PlayTennis'
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(sample_data, features, label)

    instance = {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': 'False'}
    prediction = tree.predict(instance)
    assert prediction == 'Yes', f"Expected 'Yes' but got {prediction}"

def test_tree_serialization(sample_data):
    features = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    label = 'PlayTennis'
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(sample_data, features, label)

    tree_json = tree.to_json()
    assert '"tree": {' in tree_json, "JSON serialization does not include 'tree' key"
