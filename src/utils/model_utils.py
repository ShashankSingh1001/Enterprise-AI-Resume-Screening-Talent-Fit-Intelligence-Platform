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
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.logging import get_logger
from src.exceptions import ModelPredictionError, FileProcessingError

logger = get_logger(__name__)

# Supported model file extensions
SUPPORTED_MODEL_FORMATS = ['.pkl', '.pickle', '.joblib', '.json']


def save_model(
    model: Any, 
    model_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    create_dir: bool = True
) -> bool:
    """
    Save model to file with optional metadata.
    
    Args:
        model: Model object to save
        model_path: Path to save the model
        metadata: Optional metadata dictionary
        create_dir: Create directory if it doesn't exist
        
    Returns:
        True if successful
        
    Raises:
        FileProcessingError: If save operation fails
    """
    try:
        logger.info(f"Saving model to: {model_path}")
        
        # Create directory if needed
        if create_dir:
            model_dir = os.path.dirname(model_path)
            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {model_dir}")
        
        # Determine file format
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext not in SUPPORTED_MODEL_FORMATS:
            raise FileProcessingError(
                Exception(
                    f"Unsupported model format: {file_ext}. "
                    f"Supported: {SUPPORTED_MODEL_FORMATS}"
                ),
                sys
            )
        
        # Save model based on format
        if file_ext in ['.pkl', '.pickle']:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif file_ext == '.joblib':
            joblib.dump(model, model_path)
        elif file_ext == '.json':
            with open(model_path, 'w') as f:
                json.dump(model, f, indent=2)
        
        logger.info(f"Model saved successfully: {model_path}")
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = model_path.replace(file_ext, '_metadata.json')
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['model_path'] = model_path
            metadata['file_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata saved: {metadata_path}")
        
        return True
    
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise FileProcessingError(e, sys)


def load_model(model_path: str) -> Any:
    """
    Load model from file.
    Supports .pkl, .pickle, .joblib, .json formats.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileProcessingError: If load operation fails
    """
    try:
        logger.info(f"Loading model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileProcessingError(
                Exception(f"Model file not found: {model_path}"),
                sys
            )
        
        # Determine file format
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext not in SUPPORTED_MODEL_FORMATS:
            raise FileProcessingError(
                Exception(
                    f"Unsupported model format: {file_ext}. "
                    f"Supported: {SUPPORTED_MODEL_FORMATS}"
                ),
                sys
            )
        
        # Load model based on format
        if file_ext in ['.pkl', '.pickle']:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif file_ext == '.joblib':
            model = joblib.load(model_path)
        elif file_ext == '.json':
            with open(model_path, 'r') as f:
                model = json.load(f)
        
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        return model
    
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise FileProcessingError(e, sys)


def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Extract model metadata including file info.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model metadata
        
    Raises:
        FileProcessingError: If metadata extraction fails
    """
    try:
        logger.debug(f"Extracting metadata for: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileProcessingError(
                Exception(f"Model file not found: {model_path}"),
                sys
            )
        
        # Basic file metadata
        stat_info = os.stat(model_path)
        file_ext = os.path.splitext(model_path)[1]
        
        metadata = {
            'model_path': model_path,
            'filename': os.path.basename(model_path),
            'file_format': file_ext,
            'size_bytes': stat_info.st_size,
            'size_mb': round(stat_info.st_size / (1024 * 1024), 2),
            'created_at': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
        }
        
        # Try to load saved metadata file
        metadata_path = model_path.replace(file_ext, '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
                metadata.update(saved_metadata)
            logger.debug("Loaded saved metadata file")
        
        logger.info(f"Metadata extracted: {metadata['size_mb']}MB")
        return metadata
    
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"Failed to get model metadata: {str(e)}")
        raise FileProcessingError(e, sys)


def validate_model_input(
    X: np.ndarray,
    expected_features: Optional[int] = None,
    check_nan: bool = True,
    check_inf: bool = True
) -> bool:
    """
    Validate model input data.
    
    Args:
        X: Input features array
        expected_features: Expected number of features
        check_nan: Check for NaN values
        check_inf: Check for infinite values
        
    Returns:
        True if valid
        
    Raises:
        ModelPredictionError: If validation fails
    """
    try:
        logger.debug(f"Validating model input: shape={X.shape}")
        
        # Check if numpy array
        if not isinstance(X, np.ndarray):
            raise ModelPredictionError(
                Exception(f"Input must be numpy array, got: {type(X).__name__}"),
                sys
            )
        
        # Check dimensions
        if X.ndim not in [1, 2]:
            raise ModelPredictionError(
                Exception(f"Input must be 1D or 2D array, got: {X.ndim}D"),
                sys
            )
        
        # Check expected features
        if expected_features is not None:
            n_features = X.shape[1] if X.ndim == 2 else X.shape[0]
            if n_features != expected_features:
                raise ModelPredictionError(
                    Exception(
                        f"Expected {expected_features} features, got {n_features}"
                    ),
                    sys
                )
        
        # Check for NaN values
        if check_nan and np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            raise ModelPredictionError(
                Exception(f"Input contains {nan_count} NaN values"),
                sys
            )
        
        # Check for infinite values
        if check_inf and np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            raise ModelPredictionError(
                Exception(f"Input contains {inf_count} infinite values"),
                sys
            )
        
        logger.debug("Model input validation successful")
        return True
    
    except ModelPredictionError:
        raise
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise ModelPredictionError(e, sys)


def preprocess_features(
    features: Dict[str, Any],
    feature_names: List[str],
    scaler: Optional[StandardScaler] = None,
    fill_missing: bool = True
) -> np.ndarray:
    """
    Convert feature dictionary to numpy array for model input.
    
    Args:
        features: Dictionary of feature values
        feature_names: Ordered list of feature names
        scaler: Optional StandardScaler for normalization
        fill_missing: Fill missing values with 0
        
    Returns:
        Numpy array ready for model prediction
        
    Raises:
        ModelPredictionError: If preprocessing fails
    """
    try:
        logger.debug(f"Preprocessing {len(features)} features")
        
        # Extract values in correct order
        values = []
        for name in feature_names:
            if name in features:
                values.append(features[name])
            elif fill_missing:
                logger.warning(f"Missing feature '{name}', filling with 0")
                values.append(0)
            else:
                raise ModelPredictionError(
                    Exception(f"Required feature missing: {name}"),
                    sys
                )
        
        # Convert to numpy array
        X = np.array(values).reshape(1, -1)
        
        # Apply scaling if provided
        if scaler is not None:
            X = scaler.transform(X)
            logger.debug("Features scaled")
        
        logger.info(f"Features preprocessed: shape={X.shape}")
        return X
    
    except ModelPredictionError:
        raise
    except Exception as e:
        logger.error(f"Feature preprocessing failed: {str(e)}")
        raise ModelPredictionError(e, sys)


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Extract and rank feature importance from model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Return only top N features (None = all)
        
    Returns:
        List of (feature_name, importance) tuples, sorted by importance
    """
    try:
        logger.debug("Calculating feature importance")
        
        # Check if model has feature importance
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {type(model).__name__} doesn't have feature_importances_")
            return []
        
        importances = model.feature_importances_
        
        # Create (name, importance) pairs
        feature_importance = list(zip(feature_names, importances))
        
        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N if specified
        if top_n is not None:
            feature_importance = feature_importance[:top_n]
        
        logger.info(f"Feature importance calculated for {len(feature_importance)} features")
        return feature_importance
    
    except Exception as e:
        logger.error(f"Feature importance calculation failed: {str(e)}")
        return []


def predict_with_validation(
    model: Any,
    X: np.ndarray,
    return_proba: bool = False
) -> np.ndarray:
    """
    Make predictions with input validation.
    
    Args:
        model: Trained model
        X: Input features
        return_proba: Return probabilities instead of class predictions
        
    Returns:
        Predictions or probabilities
        
    Raises:
        ModelPredictionError: If prediction fails
    """
    try:
        logger.debug(f"Making prediction: shape={X.shape}")
        
        # Validate input
        validate_model_input(X)
        
        # Make prediction
        if return_proba and hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
            logger.debug("Returned class probabilities")
        else:
            predictions = model.predict(X)
            logger.debug("Returned class predictions")
        
        logger.info(f"Prediction successful: {predictions.shape}")
        return predictions
    
    except ModelPredictionError:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ModelPredictionError(e, sys)


def handle_missing_values(
    X: np.ndarray,
    strategy: str = 'mean'
) -> np.ndarray:
    """
    Handle missing values in feature array.
    
    Args:
        X: Input array with potential missing values
        strategy: Strategy for filling ('mean', 'median', 'zero')
        
    Returns:
        Array with missing values filled
    """
    try:
        if not np.isnan(X).any():
            logger.debug("No missing values found")
            return X
        
        nan_count = np.isnan(X).sum()
        logger.info(f"Handling {nan_count} missing values using '{strategy}' strategy")
        
        X_filled = X.copy()
        
        if strategy == 'mean':
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X_filled[nan_indices] = np.take(col_means, nan_indices[1])
        
        elif strategy == 'median':
            col_medians = np.nanmedian(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X_filled[nan_indices] = np.take(col_medians, nan_indices[1])
        
        elif strategy == 'zero':
            X_filled = np.nan_to_num(X, nan=0.0)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.debug("Missing values handled successfully")
        return X_filled
    
    except Exception as e:
        logger.error(f"Failed to handle missing values: {str(e)}")
        raise ModelPredictionError(e, sys)


def create_feature_dict(
    feature_names: List[str],
    values: List[Any]
) -> Dict[str, Any]:
    """
    Create feature dictionary from names and values.
    
    Args:
        feature_names: List of feature names
        values: List of feature values
        
    Returns:
        Dictionary mapping feature names to values
    """
    try:
        if len(feature_names) != len(values):
            raise ValueError(
                f"Length mismatch: {len(feature_names)} names vs {len(values)} values"
            )
        
        feature_dict = dict(zip(feature_names, values))
        logger.debug(f"Created feature dictionary with {len(feature_dict)} features")
        return feature_dict
    
    except Exception as e:
        logger.error(f"Failed to create feature dictionary: {str(e)}")
        return {}


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Extract information about a model object.
    
    Args:
        model: Model object
        
    Returns:
        Dictionary with model information
    """
    try:
        info = {
            'model_type': type(model).__name__,
            'module': type(model).__module__,
        }
        
        # Add model-specific attributes
        if hasattr(model, 'get_params'):
            info['params'] = model.get_params()
        
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        
        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()
        
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importance'] = True
        
        logger.debug(f"Model info extracted: {info['model_type']}")
        return info
    
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return {'model_type': 'unknown'}