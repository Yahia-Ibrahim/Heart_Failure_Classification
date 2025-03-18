import numpy as np
import pandas as pd

class Node:
    def __init__(self, value=None, right=None, left=None, feature=None, threshold=None, is_leaf=False):
        self.value = value
        self.right = right
        self.left = left
        self.feature = feature
        self.is_leaf = is_leaf
        self.threshold = threshold

class DecisionTreeClassifier:
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.root = None

    def fit(self, x_train, y_train, sample_weight=None):
        # Convert Pandas DataFrames/Series to NumPy arrays
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if sample_weight is not None and isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.to_numpy()
        
        # Ensure all inputs are NumPy arrays
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        if sample_weight is None:
            sample_weight = np.ones(len(y_train)) / len(y_train)  # Equal weights if not provided
        self.root = self._build_tree_recursive(x_train=x_train, y_train=y_train, sample_weight=sample_weight)


    def _build_tree_recursive(self, x_train, y_train, depth=0, sample_weight=None) -> Node:
        if sample_weight is None:
            sample_weight = np.ones_like(y_train, dtype=float)

        # Stop if max depth is reached or only one class remains
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if depth >= self.max_depth or len(unique_classes) == 1:
            leaf_value = unique_classes[np.argmax(class_counts)]  # Most frequent class
            return Node(value=leaf_value, is_leaf=True)
        
        # Use weighted class counts if sample weights are provided
        if sample_weight is not None:
            class_weights = np.zeros_like(unique_classes, dtype=float)
            for i, c in enumerate(unique_classes):
                class_weights[i] = np.sum(sample_weight[y_train == c])
            leaf_value = unique_classes[np.argmax(class_weights)]  # Most weighted class
        else:
            leaf_value = unique_classes[np.argmax(class_counts)]

        left_samples, right_samples, best_feature, best_threshold = self._split(x_train, y_train, sample_weight)

        if best_feature is None or left_samples.sum() == 0 or right_samples.sum() == 0:
            return Node(value=leaf_value, is_leaf=True)

        left = self._build_tree_recursive(
            x_train[left_samples, :], y_train[left_samples], depth + 1, sample_weight[left_samples]
        )
        right = self._build_tree_recursive(
            x_train[right_samples, :], y_train[right_samples], depth + 1, sample_weight[right_samples]
        )

        return Node(value=None, right=right, left=left, feature=best_feature, threshold=best_threshold)

    def _split(self, x_train, y_train, sample_weight):
   
        max_info_gain = -float('inf')
        best_feature, best_split = None, None
    
        for feature in range(x_train.shape[1]):
            column = x_train[:, feature]
            unique_values = np.unique(column)
    
            for split in unique_values:
                info_gain = self._information_gain(column, y_train, split, sample_weight)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature, best_split = feature, split
    
        if best_feature is None:  # Handle edge case where no split is found
            return np.array([]), np.array([]), None, None
    
        left_indices = np.where(x_train[:, best_feature] <= best_split)[0]
        right_indices = np.where(x_train[:, best_feature] > best_split)[0]
    
        return left_indices, right_indices, best_feature, best_split



    def _information_gain(self, x, y, threshold, sample_weight):
        root_entropy = self._entropy(y, sample_weight=sample_weight)

        left_indices = x <= threshold
        right_indices = x > threshold
        
        # Check if the split is valid (non-empty subsets)
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices], sample_weight=sample_weight[left_indices])
        right_entropy = self._entropy(y[right_indices], sample_weight=sample_weight[right_indices])

        # Correct weighted entropy calculation
        new_entropy = (np.sum(left_indices) / len(x)) * left_entropy + (np.sum(right_indices) / len(x)) * right_entropy

        return root_entropy - new_entropy


    def _entropy(self, data, sample_weight):
        values, counts = np.unique(data, return_counts=True)
        probabilities = np.array(
            [np.sum(sample_weight[data == v]) for v in values]
        ) / np.sum(sample_weight)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def predict(self, x):
        return np.array([self._traverse_tree(sample, self.root) for sample in x])



    def _traverse_tree(self, sample, node):
        if(self.root is None):
            raise Exception("Can't predict, Decesion Tree wasn't trained on any data")
        if(node.is_leaf):
            return node.value

        if sample[node.feature] <= node.threshold:
            return self._traverse_tree(sample=sample, node=node.left)

        return self._traverse_tree(sample=sample, node=node.right)


         