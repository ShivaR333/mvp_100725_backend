import json
import os
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import io
import re
from typing import Dict, List, Any
from decimal import Decimal

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

S3_BUCKET = os.environ['S3_BUCKET_NAME']
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
table = dynamodb.Table(DYNAMODB_TABLE)

def lambda_handler(event, context):
    """
    Lambda function to map CSV schema and infer column types
    """
    print(f"Schema mapping triggered: {json.dumps(event, default=str)}")
    
    try:
        # Extract S3 information from Step Functions input
        bucket = event.get('bucket') or S3_BUCKET
        key = event.get('key')
        
        if not key:
            raise ValueError("S3 key not provided in event")
        
        # Extract user_id and dataset_id from S3 key
        # Expected format: datasets/{user_id}/{dataset_id}/raw.csv
        key_parts = key.split('/')
        if len(key_parts) != 4:
            raise ValueError(f"Invalid S3 key format: {key}")
        
        user_id = key_parts[1]
        dataset_id = key_parts[2]
        
        print(f"Processing dataset: {dataset_id} for user: {user_id}")
        
        # Update status to processing
        update_status(user_id, dataset_id, 'processing', 'Mapping schema...')
        
        # Download CSV from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        
        # Load CSV into pandas DataFrame
        df = pd.read_csv(io.StringIO(csv_content))
        
        print(f"Loaded DataFrame with shape: {df.shape}")
        
        # Perform schema mapping
        schema_info = analyze_schema(df)
        
        # Generate data preview
        preview_info = generate_data_preview(df)
        
        # Store schema information in temporary location for next step
        schema_key = key.replace('/raw.csv', '/schema.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=schema_key,
            Body=json.dumps(schema_info, indent=2),
            ContentType='application/json'
        )
        
        # Store data preview
        preview_key = key.replace('/raw.csv', '/preview.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=preview_key,
            Body=json.dumps(preview_info, indent=2),
            ContentType='application/json'
        )
        
        # Update DynamoDB with schema info (convert floats to Decimals for DynamoDB)
        schema_info_dynamodb = convert_to_dynamodb_serializable(schema_info)
        table.update_item(
            Key={'user_id': user_id, 'dataset_id': dataset_id},
            UpdateExpression='SET schema_info = :schema, updated_at = :updated, processing_step = :step',
            ExpressionAttributeValues={
                ':schema': schema_info_dynamodb,
                ':updated': datetime.utcnow().isoformat(),
                ':step': 'schema_mapped'
            }
        )
        
        print(f"Schema mapping completed for dataset: {dataset_id}")
        
        # Return event data for next step
        return {
            'bucket': bucket,
            'key': key,
            'schema_key': schema_key,
            'user_id': user_id,
            'dataset_id': dataset_id,
            'schema_info': schema_info,
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
    except Exception as e:
        print(f"Error in schema mapping: {str(e)}")
        
        # Update status to failed if we have the dataset info
        try:
            if 'user_id' in locals() and 'dataset_id' in locals():
                update_status(user_id, dataset_id, 'failed', f'Schema mapping failed: {str(e)}')
        except:
            pass
        
        raise e

def convert_to_json_serializable(obj):
    """
    Convert numpy/pandas types to JSON serializable Python types
    """
    if pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

def convert_to_dynamodb_serializable(obj):
    """
    Convert data to DynamoDB serializable format (converts floats to Decimals)
    """
    if obj is None:
        return None
    elif isinstance(obj, float):
        if np.isfinite(obj):
            return Decimal(str(obj))  # Convert float to Decimal for DynamoDB
        else:
            return None
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_dynamodb_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dynamodb_serializable(item) for item in obj]
    else:
        return obj

def analyze_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze DataFrame schema and infer column types with statistics
    """
    schema = {
        'columns': [],
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'analysis_timestamp': datetime.utcnow().isoformat()
    }
    
    for col in df.columns:
        col_info = {
            'name': col,
            'pandas_dtype': str(df[col].dtype),
            'null_count': convert_to_json_serializable(df[col].isnull().sum()),
            'null_percentage': convert_to_json_serializable(df[col].isnull().sum() / len(df) * 100),
            'unique_count': convert_to_json_serializable(df[col].nunique()),
            'unique_percentage': convert_to_json_serializable(df[col].nunique() / len(df) * 100)
        }
        
        # Infer semantic type
        semantic_type = infer_semantic_type(df[col])
        col_info['semantic_type'] = semantic_type
        col_info['suggested_type'] = map_to_standard_type(semantic_type)
        
        # Add type-specific statistics
        if semantic_type in ['integer', 'float']:
            add_numeric_stats(df[col], col_info)
        elif semantic_type == 'datetime':
            add_datetime_stats(df[col], col_info)
        elif semantic_type in ['string', 'categorical']:
            add_text_stats(df[col], col_info)
        
        # Sample values (non-null)
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_size = min(5, len(non_null_values))
            col_info['sample_values'] = [convert_to_json_serializable(val) for val in non_null_values.head(sample_size).tolist()]
        
        schema['columns'].append(col_info)
    
    return schema

def infer_semantic_type(series: pd.Series) -> str:
    """
    Infer the semantic type of a pandas Series
    """
    # Remove null values for type inference
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return 'unknown'
    
    # Check for datetime
    if is_datetime_column(non_null):
        return 'datetime'
    
    # Check for numeric types
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return 'integer'
        else:
            return 'float'
    
    # Check if string column can be converted to numeric
    if series.dtype == 'object':
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        non_null_numeric = numeric_series.dropna()
        
        # If most values can be converted to numeric, consider it numeric
        if len(non_null_numeric) / len(non_null) > 0.8:
            if (numeric_series == numeric_series.astype(int)).all():
                return 'integer'
            else:
                return 'float'
    
    # Check for categorical (low cardinality)
    unique_ratio = series.nunique() / len(series)
    if unique_ratio < 0.1 and series.nunique() < 50:
        return 'categorical'
    
    # Check for boolean
    unique_values = series.unique()
    if len(unique_values) <= 3:  # Account for nulls
        bool_patterns = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        if any(str(val).lower() in bool_patterns for val in unique_values if pd.notna(val)):
            return 'boolean'
    
    return 'string'

def is_datetime_column(series: pd.Series) -> bool:
    """
    Check if a series contains datetime values
    """
    sample_size = min(100, len(series))
    sample = series.head(sample_size)
    
    datetime_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
    ]
    
    matches = 0
    for value in sample:
        if pd.isna(value):
            continue
        value_str = str(value)
        if any(re.search(pattern, value_str) for pattern in datetime_patterns):
            matches += 1
    
    return matches / len(sample) > 0.7

def map_to_standard_type(semantic_type: str) -> str:
    """
    Map semantic type to standard SQL/JSON types
    """
    mapping = {
        'integer': 'INTEGER',
        'float': 'DECIMAL',
        'string': 'VARCHAR',
        'datetime': 'TIMESTAMP',
        'boolean': 'BOOLEAN',
        'categorical': 'VARCHAR',
        'unknown': 'VARCHAR'
    }
    return mapping.get(semantic_type, 'VARCHAR')

def add_numeric_stats(series: pd.Series, col_info: Dict):
    """Add statistics for numeric columns"""
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        col_info['min'] = convert_to_json_serializable(numeric_series.min())
        col_info['max'] = convert_to_json_serializable(numeric_series.max())
        col_info['mean'] = convert_to_json_serializable(numeric_series.mean())
        col_info['median'] = convert_to_json_serializable(numeric_series.median())
        col_info['std'] = convert_to_json_serializable(numeric_series.std())
        col_info['q25'] = convert_to_json_serializable(numeric_series.quantile(0.25))
        col_info['q75'] = convert_to_json_serializable(numeric_series.quantile(0.75))
    except:
        pass

def add_datetime_stats(series: pd.Series, col_info: Dict):
    """Add statistics for datetime columns"""
    try:
        dt_series = pd.to_datetime(series, errors='coerce')
        col_info['min_date'] = convert_to_json_serializable(dt_series.min())
        col_info['max_date'] = convert_to_json_serializable(dt_series.max())
        col_info['date_range_days'] = convert_to_json_serializable((dt_series.max() - dt_series.min()).days) if pd.notna(dt_series.min()) and pd.notna(dt_series.max()) else None
    except:
        pass

def add_text_stats(series: pd.Series, col_info: Dict):
    """Add statistics for text columns"""
    try:
        text_lengths = series.astype(str).str.len()
        col_info['avg_length'] = convert_to_json_serializable(text_lengths.mean())
        col_info['min_length'] = convert_to_json_serializable(text_lengths.min())
        col_info['max_length'] = convert_to_json_serializable(text_lengths.max())
        
        # Most common values
        value_counts = series.value_counts().head(5)
        col_info['top_values'] = [
            {'value': str(val), 'count': convert_to_json_serializable(count)} 
            for val, count in value_counts.items()
        ]
    except:
        pass

def generate_data_preview(df: pd.DataFrame, num_rows: int = 10) -> Dict[str, Any]:
    """Generate a preview of the dataset with sample rows"""
    preview = {
        'raw_data': {
            'sample_rows': [],
            'total_rows': convert_to_json_serializable(len(df)),
            'total_columns': convert_to_json_serializable(len(df.columns)),
            'columns': list(df.columns)
        },
        'data_quality': {
            'missing_values_per_column': {},
            'duplicate_rows': convert_to_json_serializable(df.duplicated().sum()),
            'empty_columns': [],
            'data_types': {}
        },
        'generated_at': datetime.utcnow().isoformat()
    }
    
    # Sample rows (convert to JSON-serializable format)
    sample_df = df.head(num_rows)
    for idx, row in sample_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            # Handle NaN and other non-serializable values
            if pd.isna(value):
                row_dict[col] = None
            elif isinstance(value, (np.integer, np.floating)):
                row_dict[col] = float(value) if np.isfinite(value) else None
            else:
                row_dict[col] = str(value)
        preview['raw_data']['sample_rows'].append(row_dict)
    
    # Data quality metrics
    for col in df.columns:
        preview['data_quality']['missing_values_per_column'][col] = convert_to_json_serializable(df[col].isnull().sum())
        preview['data_quality']['data_types'][col] = str(df[col].dtype)
        
        # Check for empty columns
        if df[col].isnull().all():
            preview['data_quality']['empty_columns'].append(col)
    
    return preview

def update_status(user_id: str, dataset_id: str, status: str, message: str = None):
    """Update dataset status in DynamoDB"""
    update_expr = 'SET #status = :status, updated_at = :updated'
    expr_values = {
        ':status': status,
        ':updated': datetime.utcnow().isoformat()
    }
    expr_names = {'#status': 'status'}
    
    if message:
        update_expr += ', processing_message = :message'
        expr_values[':message'] = message
    
    table.update_item(
        Key={'user_id': user_id, 'dataset_id': dataset_id},
        UpdateExpression=update_expr,
        ExpressionAttributeValues=expr_values,
        ExpressionAttributeNames=expr_names
    )