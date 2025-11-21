import numpy as np
from typing import List, Dict
import logging
from .classifier import RailwayDefectClassifier

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    def __init__(self, models: List[Dict] = None):
        """
        Initialize ensemble classifier

        Args:
            models: List of model configurations
                   [{'type': 'random_forest', 'params': {...}}, ...]
        """
        self.models = []
        self.weights = []

        if models is None:
            # Default ensemble: RF + XGBoost + SVM
            models = [
                {'type': 'random_forest', 'params': {'n_estimators': 100, 'max_depth': 20}},
                {'type': 'xgboost', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
                {'type': 'svm', 'params': {'kernel': 'rbf', 'C': 1.0}}
            ]

        # Initialize models
        for model_config in models:
            model = RailwayDefectClassifier(
                model_type=model_config['type'],
                **model_config.get('params', {})
            )
            self.models.append(model)

        # Equal weights initially
        self.weights = [1.0 / len(self.models)] * len(self.models)

        logger.info(f"Initialized ensemble with {len(self.models)} models")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train all models in ensemble

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training ensemble of {len(self.models)} models...")

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_type}")
            model.train(X_train, y_train)

        logger.info("Ensemble training completed")

    def predict(self, X_test: np.ndarray, method: str = 'weighted_vote') -> np.ndarray:
        """
        Make predictions using ensemble

        Args:
            X_test: Test features
            method: Ensemble method ('weighted_vote', 'majority_vote', 'average_proba')

        Returns:
            Predicted labels
        """
        if method == 'majority_vote':
            return self._majority_vote(X_test)
        elif method == 'weighted_vote':
            return self._weighted_vote(X_test)
        elif method == 'average_proba':
            return self._average_probability(X_test)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def _majority_vote(self, X_test: np.ndarray) -> np.ndarray:
        """Majority voting"""
        predictions = np.array([model.predict(X_test) for model in self.models])

        # Get most common prediction for each sample
        final_predictions = []
        for i in range(X_test.shape[0]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])

        return np.array(final_predictions)

    def _weighted_vote(self, X_test: np.ndarray) -> np.ndarray:
        """Weighted voting based on model weights"""
        predictions = np.array([model.predict(X_test) for model in self.models])

        final_predictions = []
        for i in range(X_test.shape[0]):
            votes = predictions[:, i]

            # Count weighted votes
            vote_counts = {}
            for j, vote in enumerate(votes):
                vote_counts[vote] = vote_counts.get(vote, 0) + self.weights[j]

            # Get prediction with highest weight
            final_predictions.append(max(vote_counts, key=vote_counts.get))

        return np.array(final_predictions)

    def _average_probability(self, X_test: np.ndarray) -> np.ndarray:
        """Average probability prediction"""
        probabilities = []

        for model in self.models:
            proba = model.predict_proba(X_test)
            probabilities.append(proba)

        # Average probabilities
        avg_proba = np.mean(probabilities, axis=0)

        # Get class with highest average probability
        predictions = np.argmax(avg_proba, axis=1)

        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get ensemble probability predictions

        Args:
            X_test: Test features

        Returns:
            Average probability across all models
        """
        probabilities = []

        for i, model in enumerate(self.models):
            proba = model.predict_proba(X_test)
            probabilities.append(proba * self.weights[i])

        # Weighted average
        avg_proba = np.sum(probabilities, axis=0)

        # Normalize
        row_sums = avg_proba.sum(axis=1, keepdims=True)
        avg_proba = avg_proba / row_sums

        return avg_proba

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate ensemble performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        predictions = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }

        logger.info(f"Ensemble Evaluation: {metrics}")
        return metrics

    def set_weights(self, weights: List[float]):
        """
        Set model weights

        Args:
            weights: List of weights (must sum to 1.0)
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        self.weights = weights
        logger.info(f"Updated model weights: {weights}")