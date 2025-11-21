
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class RailwayDefectClassifier:
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize classifier

        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'xgboost')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None

        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=kwargs.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized {model_type} classifier")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'RailwayDefectClassifier':
        """
        Train the classifier

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self for method chaining
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        logger.info(f"Training {self.model_type} on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        logger.info("Training completed")

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X_test: Test features

        Returns:
            Predicted labels
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X_test: Test features

        Returns:
            Class probabilities
        """
        X_test_scaled = self.scaler.transform(X_test)
        probabilities = self.model.predict_proba(X_test_scaled)
        return probabilities

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (for tree-based models)

        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning(f"{self.model_type} does not support feature importance")
            return None

    def save(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        logger.info(f"Model loaded from {filepath}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }

        logger.info(f"Model Evaluation: {metrics}")
        return metrics