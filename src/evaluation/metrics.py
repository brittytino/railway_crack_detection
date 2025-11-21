
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, class_labels: dict = None):
        """
        Initialize Model Evaluator

        Args:
            class_labels: Dictionary mapping class indices to labels
        """
        self.class_labels = class_labels or {0: 'Healthy', 1: 'Defective'}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: np.ndarray = None) -> dict:
        """
        Comprehensive model evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # ROC-AUC (for binary or multi-class with probabilities)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, 
                                                       multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
                metrics['roc_auc'] = None

        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=[self.class_labels.get(i, f"Class_{i}") 
                         for i in range(len(np.unique(y_true)))],
            zero_division=0
        )

        logger.info(f"Model Evaluation Complete: Accuracy={metrics['accuracy']:.4f}")

        return metrics

    def calculate_confusion_matrix(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def calculate_class_metrics(self, y_true: np.ndarray, 
                                y_pred: np.ndarray) -> dict:
        """
        Calculate per-class metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of per-class metrics
        """
        unique_classes = np.unique(y_true)
        class_metrics = {}

        for cls in unique_classes:
            class_name = self.class_labels.get(cls, f"Class_{cls}")

            # Binary mask for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            class_metrics[class_name] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': np.sum(y_true == cls)
            }

        return class_metrics

    def print_evaluation_report(self, metrics: dict):
        """
        Print formatted evaluation report

        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")

        if metrics.get('roc_auc') is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        print(metrics['confusion_matrix'])

        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        print(metrics['classification_report'])
        print("="*60 + "\n")