import json
import numpy as np
import pandas as pd
from typing import Dict, Optional


class TreeNode:
    def __init__(self, feature: Optional[str] = None, label: Optional[str] = None,
                 is_root: bool = False):
        """
        A node in the decision tree

        Args:
            feature: The feature used for splitting (None for leaves)
            label: Class label (None for internal nodes)
            is_root: Or isn't
        """
        self.feature = feature
        self.label = label
        self.is_root = is_root  # is root is root or isnt root ?
        self.is_leaf = label is not None
        self.children: Dict[str, 'TreeNode'] = {}  # children of current feature, keys are values
        self.majority_class: Optional[str] = None
        self.entropy: Optional[float] = None
        self.split_gain: Optional[float] = None
        self.class_distribution: Optional[Dict[str, float]] = None
        self.all_gains: Dict[str, float] = {} # gains for all possible features

    def add_child(self, value: str, node: 'TreeNode'):
        self.children[value] = node

    def to_dict(self) -> Dict:
        base_dict = {
            'type': 'leaf' if self.is_leaf else 'node',
            'all_gains': {k: round(v, 3) for k, v in self.all_gains.items()},
            'feature': self.feature,
            'label': self.label,
            'is_root': self.is_root,
            'entropy': round(float(self.entropy), 3) if self.entropy is not None else None,
            'majority_class': self.majority_class,
            'class_distribution': self.class_distribution,
            'children': {}
        }

        if not self.is_root and not self.is_leaf:
            base_dict['gain'] = float(self.split_gain) if self.split_gain is not None else None

        for value, child in self.children.items():
            base_dict['children'][str(value)] = child.to_dict()
        #
        # if not self.is_leaf:
        #     base_dict['all_gains'] = {k: round(v, 3) for k, v in self.all_gains.items()}

        return base_dict

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


class DecisionTree():
    def __init__(self, max_depth: Optional[int] = None,  min_samples_split: int = 1):
        """
        Decision Tree classifier using ID3 algorithm.

        Args:

        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root: Optional[TreeNode] = None

    def fit(self, df: pd.DataFrame, features: list, label: str):
        self.root = self._grow_tree(
            df=df,
            features=features,
            label=label,
            depth=0,
            parent_entropy=None
        )

    def _grow_tree(self, df: pd.DataFrame, features: list, label: str,
                   depth: int, parent_entropy: Optional[float]) -> TreeNode:

        current_entropy = self._entropy(df[label])

        if df.empty:
            return TreeNode(label=None)

        if df[label].nunique() == 1:  # ol szamples bilong to one klasse
            node = TreeNode(label=df[label].iloc[0], is_root=(depth == 0))
            node.entropy = 0.0
            node.class_distribution = df[label].value_counts(normalize=False).to_dict()
            node.majority_class = majority_class(df[label])
            return node

        if not features or (self.max_depth and depth >= self.max_depth):
            node = TreeNode(label=majority_class(df[label]), is_root=(depth == 0))
            node.entropy = current_entropy
            node.class_distribution = df[label].value_counts(normalize=False).to_dict()
            node.majority_class = majority_class(df[label])
            return node

        if len(df) < self.min_samples_split:
            node = TreeNode(label=majority_class(df[label]), is_root=(depth == 0))
            node.entropy = current_entropy
            node.class_distribution = df[label].value_counts(normalize=False).to_dict()
            node.majority_class = majority_class(df[label])
            return node

        # Calculate gains for all features
        gains = {f: self._information_gain(df, f, label) for f in features}
        best_feature = max(gains, key=gains.get)

        # Create the splitting node
        node = TreeNode(feature=best_feature, is_root=(depth == 0))
        node.all_gains = gains
        #print(node.all_gains)
        node.entropy = current_entropy
        node.majority_class = majority_class(df[label])
        node.class_distribution = df[label].value_counts(normalize=False).to_dict()
        node.split_gain = gains[best_feature]

        for value, subset in df.groupby(best_feature):
            child = self._grow_tree(
                df=subset,
                features=[f for f in features if f != best_feature],
                label=label,
                depth=depth + 1,
                parent_entropy=current_entropy
            )
            node.add_child(str(value), child)

        return node

    def _select_best_feature(self, df: pd.DataFrame,
                             features: list, label: str) -> str:

        gains = {f: self._information_gain(df, f, label) for f in features}
        return max(gains, key=gains.get)

    def _information_gain(self, df: pd.DataFrame,
                          feature: str, label: str) -> float:

        base_ent = self._entropy(df[label])
        total = len(df)
        weighted_ent = 0.0

        for val, subset in df.groupby(feature):
            weighted_ent += (len(subset) / total) * self._entropy(subset[label])

        return base_ent - weighted_ent

    def predict(self, instance) -> str:
        """
        Parameters:
        - instance (dict): A dictionary representing the features of the instance to be classified.

        Returns:
        - str: The predicted class label.
        """
        if not self.root:
            raise ValueError("Prediction't. No tree was found.")

        current_node = self.root

        while not current_node.is_leaf:
            feature = current_node.feature

            if feature in instance:
                feature_value = str(instance[feature])

                if feature_value in current_node.children:
                    current_node = current_node.children[feature_value]
                else:

                    return current_node.majority_class

        #print(current_node.majority_class)
        return current_node.majority_class

    @staticmethod
    def _entropy(series: pd.Series) -> float:
        counts = series.value_counts(normalize=True)
        #print(counts)
        return -np.sum(counts * np.log2(counts))

    def to_json(self) -> str:
        """Serialize the entire tree to son of Jay"""
        if not self.root:
            raise ValueError("Tree not fitted yet")
        return json.dumps({'tree': self.root.to_dict(), 'metadata': {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split
        }}, indent=2)

    def __repr__(self):
        return self.to_json()


def majority_class(series: pd.Series) -> str:
    return series.value_counts().idxmax()


