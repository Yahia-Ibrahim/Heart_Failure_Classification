import numpy as np


class Node:
    def __init__(self, value=None, right=None, left=None, feature=None, threshold=None, is_leaf=False):
        self.value = value
        self.right = right
        self.left = left
        self.feature = feature
        self.is_leaf = is_leaf
        self.threshold = threshold

class DecisionTreeClassfier:
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.root = None

    def fit(self, x_train, y_train):
        self.root = self._build_tree_recursive(x_train=x_train, y_train=y_train)


    def _build_tree_recursive(self, x_train, y_train, depth=0) -> Node:
        # print(depth)
        if(depth >= self.max_depth or len(np.unique(y_train)) == 1):
            leaf_value = np.bincount(y_train).argmax()
            return Node(value=leaf_value, is_leaf=True)
        left_samples, right_samples, best_feature, best_threshold = self._split(x_train, y_train)
        # print(len(left_samples), " ", len(right_samples))
        # If either side is empty, stop recursion and return a leaf node
        if best_feature is None or left_samples.sum() == 0 or right_samples.sum() == 0:
            leaf_value = np.bincount(y_train).argmax()
            return Node(value=leaf_value, is_leaf=True)
        
        left = self._build_tree_recursive(x_train=x_train.iloc[left_samples, :], y_train=y_train.iloc[left_samples], depth=depth+1)
        right = self._build_tree_recursive(x_train=x_train.iloc[right_samples, :], y_train=y_train.iloc[right_samples], depth=depth+1)

        return Node(value=None, right=right, left=left, feature=best_feature, threshold=best_threshold)


    def _split(self, x_train, y_train):
        x_train = x_train.values  # Convert DataFrame to NumPy
        y_train = y_train.values
    
        max_info_gain = -float('inf')
        best_feature, best_split = None, None
    
        for feature in range(x_train.shape[1]):
            column = x_train[:, feature]
            unique_values = np.unique(column)
    
            for split in unique_values:
                info_gain = self._information_gain(column, y_train, split)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature, best_split = feature, split
    
        if best_feature is None:  # Handle edge case where no split is found
            return np.array([]), np.array([]), None, None
    
        left_indices = np.where(x_train[:, best_feature] <= best_split)[0]
        right_indices = np.where(x_train[:, best_feature] > best_split)[0]
    
        return left_indices, right_indices, best_feature, best_split



    def _information_gain(self, x, y, threshold):
        root_entropy = self._entropy(y)

        left_indices = x <= threshold
        right_indices = x > threshold
        
        # Check if the split is valid (non-empty subsets)
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        # Correct weighted entropy calculation
        new_entropy = (np.sum(left_indices) / len(x)) * left_entropy + (np.sum(right_indices) / len(x)) * right_entropy

        return root_entropy - new_entropy


    def _entropy(self, data):
        counts = np.bincount(data)
        probabilities = counts / np.sum(counts)  # Avoid zero division
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def predict(self, x):
        return np.array([self._traverse_tree(sample, self.root) for _, sample in x.iterrows()])


    def _traverse_tree(self, sample, node):
        if(self.root is None):
            raise Exception("Can't predict, Decesion Tree wasn't trained on any data")
        if(node.is_leaf):
            return node.value

        if sample.iloc[node.feature] <= node.threshold:
            return self._traverse_tree(sample=sample, node=node.left)

        return self._traverse_tree(sample=sample, node=node.right)


         