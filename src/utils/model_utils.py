"""
Model Utilities for Resume AI Platform
Provides ML model loading, saving, validation, and preprocessing functions
"""

import os
import sys
import pickle
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.logging import get_logger
from src.exceptions import ModelPredictionError, FileProcessingError

logger = get_logger(__name__)

SUPPORTED_MODEL_FORMATS = {".pkl", ".pickle", ".joblib", ".json"}


# =====================================================
# MODEL SAVE / LOAD
# =====================================================

def save_model(
    model: Any,
    model_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    create_dir: bool = True
) -> bool:
    """
    Save model to disk with optional metadata.
    """
    try:
        model_path = Path(model_path)

        if create_dir:
            model_path.parent.mkdir(parents=True, exist_ok=True)

        ext = model_path.suffix.lower()
        if ext not in SUPPORTED_MODEL_FORMATS:
            raise FileProcessingError(
                message=f"Unsupported model format: {ext}",
                error_detail=None
            )

        logger.info(f"Saving model to {model_path}")

        if ext in {".pkl", ".pickle"}:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        elif ext == ".joblib":
            joblib.dump(model, model_path)

        elif ext == ".json":
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(model, f, indent=2)

        if metadata is not None:
            metadata_path = model_path.with_name(
                model_path.stem + "_metadata.json"
            )

            metadata.update({
                "saved_at": datetime.utcnow().isoformat(),
                "model_path": str(model_path),
                "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2)
            })

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved: {metadata_path}")

        return True

    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Model save failed: {e}")
        raise FileProcessingError(
            message="Failed to save model",
            error_detail=sys.exc_info()
        )


def load_model(model_path: str) -> Any:
    """
    Load model from disk.
    """
    try:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileProcessingError(
                message=f"Model file not found: {model_path}",
                error_detail=None
            )

        ext = model_path.suffix.lower()
        if ext not in SUPPORTED_MODEL_FORMATS:
            raise FileProcessingError(
                message=f"Unsupported model format: {ext}",
                error_detail=None
            )

        logger.info(f"Loading model from {model_path}")

        if ext in {".pkl", ".pickle"}:
            with open(model_path, "rb") as f:
                return pickle.load(f)

        if ext == ".joblib":
            return joblib.load(model_path)

        if ext == ".json":
            with open(model_path, "r", encoding="utf-8") as f:
                return json.load(f)

    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise FileProcessingError(
            message="Failed to load model",
            error_detail=sys.exc_info()
        )


def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Retrieve model metadata.
    """
    try:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileProcessingError(
                message=f"Model file not found: {model_path}",
                error_detail=None
            )

        stat = model_path.stat()

        metadata = {
            "filename": model_path.name,
            "path": str(model_path),
            "format": model_path.suffix,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

        meta_file = model_path.with_name(model_path.stem + "_metadata.json")
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata.update(json.load(f))

        return metadata

    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        raise FileProcessingError(
            message="Failed to extract model metadata",
            error_detail=sys.exc_info()
        )


# =====================================================
# VALIDATION & PREPROCESSING
# =====================================================

def validate_model_input(
    X: np.ndarray,
    expected_features: Optional[int] = None,
    check_nan: bool = True,
    check_inf: bool = True
) -> bool:
    """
    Validate model input array.
    """
    try:
        if not isinstance(X, np.ndarray):
            raise ModelPredictionError(
                message="Input must be a numpy array",
                error_detail=None
            )

        if X.ndim not in (1, 2):
            raise ModelPredictionError(
                message=f"Invalid array dimensions: {X.ndim}",
                error_detail=None
            )

        if expected_features is not None:
            n_features = X.shape[-1]
            if n_features != expected_features:
                raise ModelPredictionError(
                    message=f"Expected {expected_features} features, got {n_features}",
                    error_detail=None
                )

        if check_nan and np.isnan(X).any():
            raise ModelPredictionError(
                message="Input contains NaN values",
                error_detail=None
            )

        if check_inf and np.isinf(X).any():
            raise ModelPredictionError(
                message="Input contains infinite values",
                error_detail=None
            )

        return True

    except ModelPredictionError:
        raise
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise ModelPredictionError(
            message="Model input validation failed",
            error_detail=sys.exc_info()
        )


def preprocess_features(
    features: Dict[str, Any],
    feature_names: List[str],
    scaler: Optional[StandardScaler] = None,
    fill_missing: bool = True
) -> np.ndarray:
    """
    Convert feature dictionary to numpy array.
    """
    try:
        values = []

        for name in feature_names:
            if name in features:
                values.append(features[name])
            elif fill_missing:
                values.append(0)
            else:
                raise ModelPredictionError(
                    message=f"Missing required feature: {name}",
                    error_detail=None
                )

        X = np.array(values, dtype=float).reshape(1, -1)

        if scaler is not None:
            X = scaler.transform(X)

        return X

    except ModelPredictionError:
        raise
    except Exception as e:
        logger.error(f"Feature preprocessing failed: {e}")
        raise ModelPredictionError(
            message="Feature preprocessing failed",
            error_detail=sys.exc_info()
        )


# =====================================================
# PREDICTION & ANALYSIS
# =====================================================

def predict_with_validation(
    model: Any,
    X: np.ndarray,
    return_proba: bool = False
) -> np.ndarray:
    """
    Predict with validation.
    """
    try:
        validate_model_input(X)

        if return_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(X)

        return model.predict(X)

    except ModelPredictionError:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ModelPredictionError(
            message="Model prediction failed",
            error_detail=sys.exc_info()
        )


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Extract feature importance.
    """
    if not hasattr(model, "feature_importances_"):
        return []

    importance = list(zip(feature_names, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)

    return importance[:top_n] if top_n else importance


def handle_missing_values(
    X: np.ndarray,
    strategy: str = "mean"
) -> np.ndarray:
    """
    Fill missing values.
    """
    try:
        if not np.isnan(X).any():
            return X

        X = X.copy()

        if strategy == "mean":
            means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(means, inds[1])

        elif strategy == "median":
            medians = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(medians, inds[1])

        elif strategy == "zero":
            X = np.nan_to_num(X)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return X

    except Exception as e:
        logger.error(f"Missing value handling failed: {e}")
        raise ModelPredictionError(
            message="Failed to handle missing values",
            error_detail=sys.exc_info()
        )


# =====================================================
# HELPERS
# =====================================================

def create_feature_dict(
    feature_names: List[str],
    values: List[Any]
) -> Dict[str, Any]:
    if len(feature_names) != len(values):
        return {}
    return dict(zip(feature_names, values))


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Extract model metadata safely.
    """
    info = {
        "model_type": type(model).__name__,
        "module": type(model).__module__,
    }

    if hasattr(model, "n_features_in_"):
        info["n_features"] = model.n_features_in_

    if hasattr(model, "classes_"):
        info["classes"] = list(model.classes_)

    if hasattr(model, "feature_importances_"):
        info["has_feature_importance"] = True

    if hasattr(model, "get_params"):
        info["params"] = model.get_params()

    return info
