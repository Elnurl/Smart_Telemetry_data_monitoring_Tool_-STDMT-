"""Live DataProcessor extracted from main.py (Slice C Wave 1)."""
from __future__ import annotations

import logging
import os
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller

from app.ingestion.data_reader import TIMESTAMP_COLUMN_NAMES, DataReader
from app.monitoring.observability import OBSERVABILITY

logger = logging.getLogger("SecureAnomalyDetection")


def _detect_timestamp_column(columns):
    for col in columns:
        if str(col).strip().lower() in TIMESTAMP_COLUMN_NAMES:
            return col
    return None




class DataProcessor:
    def __init__(self):
        self.data = None
        self.preprocessed_data = None
        self.feature_columns = []
        self.timestamp_column = None
        
    def load_csv(self, filepath):
        """Load data from CSV file"""
        load_span = OBSERVABILITY.start_span("data.load_csv", {"filepath": os.path.basename(str(filepath))}) if OBSERVABILITY else None
        load_message = "Data loaded successfully"
        try:
            if DataReader is not None:
                success, message, result = DataReader.load_csv(filepath)
                if not success or result is None:
                    raise ValueError(message)
                self.data = result.dataframe
                self.original_columns = result.original_columns
                self.timestamp_column = result.timestamp_column or self.timestamp_column
                load_message = message
            else:
                self.data = pd.read_csv(filepath)
                self.original_columns = self.data.columns.tolist()
                ts_col = _detect_timestamp_column(self.original_columns)
                if ts_col:
                    self.timestamp_column = ts_col
                elif len(self.data.columns) >= 2:
                    first_col = self.original_columns[0]
                    sample = self.data[first_col].head(min(20, len(self.data)))
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        self.timestamp_column = first_col
            if OBSERVABILITY:
                row_count = int(len(self.data)) if self.data is not None else 0
                col_count = int(len(self.data.columns)) if self.data is not None else 0
                OBSERVABILITY.inc_counter("data_ingest_total", labels={"source": "csv", "status": "success"})
                OBSERVABILITY.observe("data_ingest_rows", row_count, labels={"source": "csv"})
                OBSERVABILITY.end_span(load_span, status="ok", attributes={"rows": row_count, "columns": col_count})
            return True, load_message
        except Exception as e:
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("data_ingest_total", labels={"source": "csv", "status": "error"})
                OBSERVABILITY.end_span(load_span, status="error", error=e)
            return False, f"Error loading data: {str(e)}"
    
    def load_json(self, filepath):
        """Load data from JSON file"""
        load_span = OBSERVABILITY.start_span("data.load_json", {"filepath": os.path.basename(str(filepath))}) if OBSERVABILITY else None
        try:
            if DataReader is not None:
                success, message, result = DataReader.load_json(filepath)
                if not success or result is None:
                    raise ValueError(message)
                self.data = result.dataframe
                self.original_columns = result.original_columns
                self.timestamp_column = result.timestamp_column or self.timestamp_column
            else:
                self.data = pd.read_json(filepath)
                # Rename columns to standard format ['time', 'value']
                if len(self.data.columns) >= 2:
                    original_columns = self.data.columns.tolist()
                    # Store original column names for reference
                    self.original_columns = original_columns
                    # Create a mapping from original to new column names
                    rename_map = {original_columns[0]: 'time', original_columns[1]: 'value'}
                    # If there are more columns, name them value2, value3, etc.
                    for i, col in enumerate(original_columns[2:], start=2):
                        rename_map[col] = f'value{i}'
                    # Rename the columns
                    self.data = self.data.rename(columns=rename_map)
                    # Set timestamp column
                    self.timestamp_column = 'time'
            if OBSERVABILITY:
                row_count = int(len(self.data)) if self.data is not None else 0
                col_count = int(len(self.data.columns)) if self.data is not None else 0
                OBSERVABILITY.inc_counter("data_ingest_total", labels={"source": "json", "status": "success"})
                OBSERVABILITY.observe("data_ingest_rows", row_count, labels={"source": "json"})
                OBSERVABILITY.end_span(load_span, status="ok", attributes={"rows": row_count, "columns": col_count})
            return True, "Data loaded successfully and column names standardized to ['time', 'value']"
        except Exception as e:
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("data_ingest_total", labels={"source": "json", "status": "error"})
                OBSERVABILITY.end_span(load_span, status="error", error=e)
            return False, f"Error loading data: {str(e)}"
    
    def preprocess_data(self, timestamp_col=None, feature_cols=None, 
                       normalize=False, remove_outliers=False):
        """Preprocess the loaded data"""
        try:
            if self.data is None:
                return False, "No data loaded"
            
            df = self.data.copy()
            
            # Handle timestamp
            if timestamp_col is not None and timestamp_col in df.columns:
                self.timestamp_column = timestamp_col
                if pd.api.types.is_string_dtype(df[timestamp_col]):
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df.sort_values(by=timestamp_col, inplace=True)
            
            # Select features - explicitly exclude the timestamp column from features
            if feature_cols is not None and len(feature_cols) > 0:
                # Make sure we're not including the timestamp in feature columns
                if timestamp_col in feature_cols:
                    feature_cols = [col for col in feature_cols if col != timestamp_col]
                
                self.feature_columns = feature_cols
                features_df = df[feature_cols]
            else:
                # Auto-select numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if timestamp_col in numeric_cols:
                    numeric_cols.remove(timestamp_col)
                self.feature_columns = numeric_cols
                features_df = df[numeric_cols]
            
            # Handle missing values
            features_df.fillna(features_df.mean(), inplace=True)
            
            # Normalize if requested
            if normalize:
                scaler = StandardScaler()
                features_df = pd.DataFrame(
                    scaler.fit_transform(features_df),
                    columns=features_df.columns
                )
            
            # Remove outliers if requested
            if remove_outliers:
                z_scores = np.abs((features_df - features_df.mean()) / features_df.std())
                filtered_entries = (z_scores < 3).all(axis=1)
                features_df = features_df[filtered_entries]
                if timestamp_col is not None:
                    df = df[filtered_entries]
            
            # Make sure we're only selecting valid numeric columns for preprocessing
            numeric_features = features_df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_features) != len(features_df.columns):
                non_numeric = [col for col in features_df.columns if col not in numeric_features]
                logger.warning(f"Non-numeric columns excluded from features: {non_numeric}")
                features_df = features_df[numeric_features]
                self.feature_columns = numeric_features
            
            # Prepare final preprocessed dataframe
            if timestamp_col is not None and timestamp_col in df.columns:
                self.preprocessed_data = pd.concat([df[[timestamp_col]], features_df], axis=1)
            else:
                self.preprocessed_data = features_df
                
            return True, f"Preprocessed {len(self.preprocessed_data)} rows with {len(self.feature_columns)} features"
            
        except Exception as e:
            return False, f"Error during preprocessing: {str(e)}"
            
    def get_feature_stats(self):
        """Get basic statistics for each feature"""
        if self.data is None:
            return None
        
        stats = {}
        for col in self.feature_columns:
            if col in self.data:
                col_data = self.data[col]
                stats[col] = {
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "nulls": col_data.isnull().sum()
                }
        return stats

    def advanced_preprocess_data(self, timestamp_col=None, feature_cols=None, 
                               preprocessing_steps=None):
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            timestamp_col (str): Name of timestamp column
            feature_cols (list): List of feature columns to use
            preprocessing_steps (dict): Dictionary of preprocessing steps to apply
                Possible steps:
                - normalize (bool): Standardize features
                - remove_outliers (bool): Remove statistical outliers
                - handle_missing (str): Strategy for missing values ('mean', 'median', 'drop')
                - encode_categorical (bool): Encode categorical variables
                - resample (str): Resample time series ('1H', '1D', etc.)
                - smoothing (str): Apply smoothing ('moving_avg', 'ewm')
                - feature_scaling (str): Scaling method ('standard', 'minmax', 'robust')
                - dimension_reduction (str): PCA or other reduction technique
        """
        try:
            if self.data is None:
                return False, "No data loaded for preprocessing"

            # Create a copy of the data
            df = self.data.copy()
            preprocessing_steps = preprocessing_steps or {}
            
            # 1. Handle timestamp column
            if timestamp_col and timestamp_col in df.columns:
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    df.sort_values(by=timestamp_col, inplace=True)
                    self.timestamp_column = timestamp_col
                    
                    # Resample if specified
                    if preprocessing_steps.get('resample'):
                        df.set_index(timestamp_col, inplace=True)
                        df = df.resample(preprocessing_steps['resample']).mean()
                        df.reset_index(inplace=True)
                except Exception as e:
                    return False, f"Error processing timestamp: {str(e)}"

            # 2. Select and validate features
            if feature_cols:
                available_cols = [col for col in feature_cols if col in df.columns]
                if not available_cols:
                    return False, "No specified features found in data"
                df = df[available_cols + ([timestamp_col] if timestamp_col else [])]
            
            # 3. Handle missing values
            missing_strategy = preprocessing_steps.get('handle_missing', 'mean')
            if missing_strategy == 'drop':
                df.dropna(inplace=True)
            elif missing_strategy in ['mean', 'median']:
                for col in df.select_dtypes(include=['number']).columns:
                    if missing_strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)

            # 4. Handle categorical variables
            if preprocessing_steps.get('encode_categorical', False):
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_columns:
                    if col != timestamp_col:
                        df = pd.get_dummies(df, columns=[col], prefix=[col])

            # 5. Feature scaling
            scaling_method = preprocessing_steps.get('feature_scaling')
            if scaling_method:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                elif scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                elif scaling_method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                self.scaler = scaler

            # 6. Remove outliers
            if preprocessing_steps.get('remove_outliers', False):
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if col != timestamp_col:
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        df = df[z_scores < 3]

            # 7. Apply smoothing
            smoothing = preprocessing_steps.get('smoothing')
            if smoothing:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if smoothing == 'moving_avg':
                    window_size = preprocessing_steps.get('window_size', 3)
                    df[numeric_cols] = df[numeric_cols].rolling(window=window_size).mean()
                elif smoothing == 'ewm':
                    alpha = preprocessing_steps.get('alpha', 0.2)
                    df[numeric_cols] = df[numeric_cols].ewm(alpha=alpha).mean()

            # 8. Dimension reduction
            if preprocessing_steps.get('dimension_reduction') == 'pca':
                n_components = preprocessing_steps.get('n_components', 0.95)
                numeric_cols = df.select_dtypes(include=['number']).columns
                pca = PCA(n_components=n_components)
                df_pca = pca.fit_transform(df[numeric_cols])
                
                # Replace numeric columns with PCA components
                df = df.drop(columns=numeric_cols)
                pca_cols = [f'PC{i+1}' for i in range(df_pca.shape[1])]
                df = pd.concat([df, pd.DataFrame(df_pca, columns=pca_cols)], axis=1)
                self.pca = pca

            # Store preprocessed data and feature columns
            self.preprocessed_data = df
            self.feature_columns = [col for col in df.columns if col != timestamp_col]
            
            # Generate preprocessing summary
            summary = {
                'original_shape': self.data.shape,
                'preprocessed_shape': df.shape,
                'features': self.feature_columns,
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_features': list(df.select_dtypes(include=['number']).columns),
                'categorical_features': list(df.select_dtypes(include=['object', 'category']).columns)
            }
            
            return True, {'message': 'Preprocessing completed successfully', 'summary': summary}

        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Error during preprocessing: {str(e)}"

    def get_preprocessing_results(self):
        """
        Generate a comprehensive analysis of preprocessing results
        
        Returns:
            dict: Dictionary containing preprocessing analysis and statistics
        """
        try:
            if self.preprocessed_data is None:
                return None, "No preprocessed data available"

            results = {
                "general_info": {
                    "original_shape": self.data.shape if self.data is not None else None,
                    "preprocessed_shape": self.preprocessed_data.shape,
                    "rows_difference": len(self.data) - len(self.preprocessed_data) if self.data is not None else 0,
                    "features_count": len(self.feature_columns) if self.feature_columns else 0,
                    "timestamp_column": self.timestamp_column
                },
                
                "data_quality": {
                    "missing_values": {
                        col: self.preprocessed_data[col].isnull().sum() 
                        for col in self.preprocessed_data.columns
                    },
                    "unique_values": {
                        col: self.preprocessed_data[col].nunique()
                        for col in self.preprocessed_data.columns
                    }
                },
                
                "numerical_stats": {
                    col: {
                        "mean": self.preprocessed_data[col].mean(),
                        "std": self.preprocessed_data[col].std(),
                        "min": self.preprocessed_data[col].min(),
                        "max": self.preprocessed_data[col].max(),
                        "q25": self.preprocessed_data[col].quantile(0.25),
                        "q50": self.preprocessed_data[col].quantile(0.50),
                        "q75": self.preprocessed_data[col].quantile(0.75),
                        "skewness": self.preprocessed_data[col].skew(),
                        "kurtosis": self.preprocessed_data[col].kurtosis()
                    }
                    for col in self.preprocessed_data.select_dtypes(include=['number']).columns
                },
                
                "temporal_stats": {} if self.timestamp_column is None else {
                    "time_range": {
                        "start": self.preprocessed_data[self.timestamp_column].min(),
                        "end": self.preprocessed_data[self.timestamp_column].max(),
                        "duration": str(self.preprocessed_data[self.timestamp_column].max() - 
                                     self.preprocessed_data[self.timestamp_column].min())
                    },
                    "frequency": self.preprocessed_data[self.timestamp_column].diff().mean()
                },
                
                "feature_correlations": self.preprocessed_data.select_dtypes(
                    include=['number']).corr().to_dict() if len(self.preprocessed_data.select_dtypes(
                    include=['number']).columns) > 1 else {},
                
                "outlier_summary": {
                    col: {
                        "outliers_count": len(self.preprocessed_data[
                            (self.preprocessed_data[col] > 
                             self.preprocessed_data[col].mean() + 3 * self.preprocessed_data[col].std()) |
                            (self.preprocessed_data[col] < 
                             self.preprocessed_data[col].mean() - 3 * self.preprocessed_data[col].std())
                        ]),
                        "outliers_percentage": (len(self.preprocessed_data[
                            (self.preprocessed_data[col] > 
                             self.preprocessed_data[col].mean() + 3 * self.preprocessed_data[col].std()) |
                            (self.preprocessed_data[col] < 
                             self.preprocessed_data[col].mean() - 3 * self.preprocessed_data[col].std())
                        ]) / len(self.preprocessed_data)) * 100
                    }
                    for col in self.preprocessed_data.select_dtypes(include=['number']).columns
                }
            }
            
            # Add data distribution analysis
            if self.preprocessed_data is not None:
                results["distribution_tests"] = {
                    col: {
                        "normality": {
                            "shapiro": stats.shapiro(
                                self.preprocessed_data[col].dropna()
                            ) if len(self.preprocessed_data[col].dropna()) >= 3 else None
                        },
                        "stationarity": {
                            "adf_test": adfuller(
                                self.preprocessed_data[col].dropna()
                            ) if len(self.preprocessed_data[col].dropna()) >= 3 else None
                        }
                    }
                    for col in self.preprocessed_data.select_dtypes(include=['number']).columns
                }
            
            return True, results

        except Exception as e:
            logger.error(f"Error generating preprocessing results: {str(e)}")
            return False, f"Error analyzing preprocessing results: {str(e)}"

    def format_preprocessing_results(self, results):
        """
        Format preprocessing results into a human-readable format
        
        Args:
            results (dict): Results from get_preprocessing_results()
            
        Returns:
            str: Formatted string containing analysis results
        """
        if not results:
            return "No preprocessing results available"

        formatted_output = []
        formatted_output.append("=== Preprocessing Analysis Report ===\n")
        
        # General Information
        formatted_output.append("General Information:")
        gen_info = results["general_info"]
        formatted_output.append(f"- Original data shape: {gen_info['original_shape']}")
        formatted_output.append(f"- Preprocessed data shape: {gen_info['preprocessed_shape']}")
        formatted_output.append(f"- Rows removed: {gen_info['rows_difference']}")
        formatted_output.append(f"- Features used: {gen_info['features_count']}")
        formatted_output.append(f"- Timestamp column: {gen_info['timestamp_column']}\n")
        
        # Data Quality
        formatted_output.append("Data Quality:")
        for col, missing in results["data_quality"]["missing_values"].items():
            if missing > 0:
                formatted_output.append(f"- {col}: {missing} missing values")
        formatted_output.append("")
        
        # Numerical Statistics
        formatted_output.append("Numerical Statistics:")
        for col, stats in results["numerical_stats"].items():
            formatted_output.append(f"\n{col}:")
            formatted_output.append(f"- Mean: {stats['mean']:.2f}")
            formatted_output.append(f"- Std: {stats['std']:.2f}")
            formatted_output.append(f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            formatted_output.append(f"- Quartiles: {stats['q25']:.2f}, {stats['q50']:.2f}, {stats['q75']:.2f}")
        
        # Temporal Statistics
        if results["temporal_stats"]:
            formatted_output.append("\nTemporal Statistics:")
            time_range = results["temporal_stats"]["time_range"]
            formatted_output.append(f"- Time range: {time_range['start']} to {time_range['end']}")
            formatted_output.append(f"- Duration: {time_range['duration']}")
            formatted_output.append(f"- Average frequency: {results['temporal_stats']['frequency']}\n")
        
        # Outlier Summary
        formatted_output.append("Outlier Summary:")
        for col, outlier_stats in results["outlier_summary"].items():
            if outlier_stats["outliers_count"] > 0:
                formatted_output.append(
                    f"- {col}: {outlier_stats['outliers_count']} outliers "
                    f"({outlier_stats['outliers_percentage']:.2f}%)"
                )
        
        return "\n".join(formatted_output)
