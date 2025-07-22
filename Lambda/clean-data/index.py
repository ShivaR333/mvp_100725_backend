import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os
from typing import Dict, List, Any, Tuple

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

S3_BUCKET = os.environ['S3_BUCKET_NAME']
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
table = dynamodb.Table(DYNAMODB_TABLE)

def lambda_handler(event, context):
    """
    Lambda function to clean data based on schema analysis
    """
    print(f"Data cleaning triggered: {json.dumps(event, default=str)}")
    
    try:
        # Extract information from previous step
        bucket = event.get('bucket') or S3_BUCKET
        key = event.get('key')
        schema_key = event.get('schema_key')
        user_id = event.get('user_id')
        dataset_id = event.get('dataset_id')
        schema_info = event.get('schema_info')
        
        if not all([key, user_id, dataset_id]):
            raise ValueError("Missing required parameters from previous step")
        
        print(f"Cleaning dataset: {dataset_id} for user: {user_id}")
        
        # Update status
        update_status(user_id, dataset_id, 'processing', 'Cleaning data...')
        
        # Load original CSV
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        
        print(f"Loaded DataFrame with shape: {df.shape}")
        
        # Load schema information if not provided
        if not schema_info:
            schema_response = s3_client.get_object(Bucket=bucket, Key=schema_key)
            schema_info = json.loads(schema_response['Body'].read().decode('utf-8'))
        
        # Perform data cleaning
        cleaned_df, cleaning_report = clean_dataframe(df, schema_info)
        
        print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
        
        # Save cleaned data as Parquet
        cleaned_key = key.replace('/raw.csv', '/cleaned.parquet')
        parquet_buffer = io.BytesIO()
        cleaned_df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_client.put_object(
            Bucket=bucket,
            Key=cleaned_key,
            Body=parquet_buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        
        # Generate cleaned data preview
        cleaned_preview = generate_cleaned_data_preview(cleaned_df, cleaning_report)
        
        # Save cleaning report
        report_key = key.replace('/raw.csv', '/cleaning_report.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=report_key,
            Body=json.dumps(cleaning_report, indent=2),
            ContentType='application/json'
        )
        
        # Save cleaned data preview
        cleaned_preview_key = key.replace('/raw.csv', '/cleaned_preview.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=cleaned_preview_key,
            Body=json.dumps(cleaned_preview, indent=2),
            ContentType='application/json'
        )
        
        print(f"Data cleaning completed for dataset: {dataset_id}")
        
        # Return event data for next step
        return {
            'bucket': bucket,
            'original_key': key,
            'cleaned_key': cleaned_key,
            'report_key': report_key,
            'schema_key': schema_key,
            'user_id': user_id,
            'dataset_id': dataset_id,
            'schema_info': schema_info,
            'cleaning_report': cleaning_report,
            'original_row_count': len(df),
            'cleaned_row_count': len(cleaned_df),
            'rows_removed': len(df) - len(cleaned_df)
        }
        
    except Exception as e:
        print(f"Error in data cleaning: {str(e)}")
        
        # Update status to failed
        try:
            if user_id and dataset_id:
                update_status(user_id, dataset_id, 'failed', f'Data cleaning failed: {str(e)}')
        except:
            pass
        
        raise e

def clean_dataframe(df: pd.DataFrame, schema_info: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the DataFrame based on schema analysis
    """
    cleaning_report = {
        'operations': [],
        'original_shape': df.shape,
        'cleaning_timestamp': datetime.utcnow().isoformat()
    }
    
    cleaned_df = df.copy()
    
    # Process each column based on its schema info
    for col_info in schema_info.get('columns', []):
        col_name = col_info['name']
        semantic_type = col_info.get('semantic_type')
        
        if col_name not in cleaned_df.columns:
            continue
        
        print(f"Cleaning column: {col_name} (type: {semantic_type})")
        
        # Apply type-specific cleaning
        if semantic_type in ['integer', 'float']:
            cleaned_df, ops = clean_numeric_column(cleaned_df, col_name, col_info)
        elif semantic_type == 'datetime':
            cleaned_df, ops = clean_datetime_column(cleaned_df, col_name, col_info)
        elif semantic_type == 'boolean':
            cleaned_df, ops = clean_boolean_column(cleaned_df, col_name, col_info)
        elif semantic_type == 'categorical':
            cleaned_df, ops = clean_categorical_column(cleaned_df, col_name, col_info)
        else:  # string
            cleaned_df, ops = clean_string_column(cleaned_df, col_name, col_info)
        
        if ops:
            cleaning_report['operations'].extend([{**op, 'column': col_name} for op in ops])
    
    # Remove rows with too many null values (optional)
    null_threshold = 0.7  # Remove rows with more than 70% nulls
    null_counts = cleaned_df.isnull().sum(axis=1)
    rows_to_remove = null_counts > (len(cleaned_df.columns) * null_threshold)
    
    if rows_to_remove.sum() > 0:
        cleaned_df = cleaned_df[~rows_to_remove]
        cleaning_report['operations'].append({
            'operation': 'remove_high_null_rows',
            'description': f'Removed {rows_to_remove.sum()} rows with >70% null values',
            'rows_affected': int(rows_to_remove.sum())
        })
    
    # Final shape and summary
    cleaning_report['final_shape'] = cleaned_df.shape
    cleaning_report['rows_removed'] = df.shape[0] - cleaned_df.shape[0]
    cleaning_report['columns_processed'] = len([col for col in schema_info.get('columns', []) if col['name'] in cleaned_df.columns])
    
    return cleaned_df, cleaning_report

def clean_numeric_column(df: pd.DataFrame, col_name: str, col_info: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """Clean numeric columns"""
    operations = []
    
    # Convert to numeric if needed
    original_dtype = df[col_name].dtype
    if original_dtype == 'object':
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        operations.append({
            'operation': 'convert_to_numeric',
            'description': f'Converted from {original_dtype} to numeric'
        })
    
    # Handle outliers using IQR method
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
    outlier_count = outliers.sum()
    
    if outlier_count > 0 and outlier_count < len(df) * 0.05:  # Less than 5% outliers
        # Cap outliers instead of removing
        df[col_name] = df[col_name].clip(lower_bound, upper_bound)
        operations.append({
            'operation': 'cap_outliers',
            'description': f'Capped {outlier_count} outliers to IQR bounds',
            'outliers_capped': int(outlier_count)
        })
    
    # Handle missing values
    null_count = df[col_name].isnull().sum()
    if null_count > 0:
        if null_count < len(df) * 0.1:  # Less than 10% missing
            # Fill with median
            median_val = df[col_name].median()
            df[col_name].fillna(median_val, inplace=True)
            operations.append({
                'operation': 'fill_missing_median',
                'description': f'Filled {null_count} missing values with median ({median_val})',
                'values_filled': int(null_count)
            })
    
    return df, operations

def clean_datetime_column(df: pd.DataFrame, col_name: str, col_info: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """Clean datetime columns"""
    operations = []
    
    # Convert to datetime
    original_dtype = df[col_name].dtype
    if original_dtype == 'object':
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        operations.append({
            'operation': 'convert_to_datetime',
            'description': f'Converted from {original_dtype} to datetime'
        })
    
    # Handle missing values
    null_count = df[col_name].isnull().sum()
    if null_count > 0 and null_count < len(df) * 0.05:  # Less than 5% missing
        # Fill with mode (most common date) or median
        if df[col_name].mode().empty:
            median_date = df[col_name].median()
            df[col_name].fillna(median_date, inplace=True)
            operations.append({
                'operation': 'fill_missing_median_date',
                'description': f'Filled {null_count} missing dates with median',
                'values_filled': int(null_count)
            })
    
    return df, operations

def clean_boolean_column(df: pd.DataFrame, col_name: str, col_info: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """Clean boolean columns"""
    operations = []
    
    # Standardize boolean values
    df[col_name] = df[col_name].astype(str).str.lower()
    
    # Map common boolean representations
    bool_mapping = {
        'true': True, '1': True, 'yes': True, 'y': True,
        'false': False, '0': False, 'no': False, 'n': False
    }
    
    df[col_name] = df[col_name].map(bool_mapping)
    operations.append({
        'operation': 'standardize_boolean',
        'description': 'Standardized boolean values to True/False'
    })
    
    # Handle missing values - fill with mode
    null_count = df[col_name].isnull().sum()
    if null_count > 0:
        mode_value = df[col_name].mode()
        if not mode_value.empty:
            df[col_name].fillna(mode_value.iloc[0], inplace=True)
            operations.append({
                'operation': 'fill_missing_mode',
                'description': f'Filled {null_count} missing values with mode',
                'values_filled': int(null_count)
            })
    
    return df, operations

def clean_categorical_column(df: pd.DataFrame, col_name: str, col_info: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """Clean categorical columns"""
    operations = []
    
    # Standardize text (trim, lowercase for comparison)
    df[col_name] = df[col_name].astype(str).str.strip()
    
    # Group similar categories (basic fuzzy matching)
    value_counts = df[col_name].value_counts()
    
    # If there are very similar values, group them
    # This is a simplified approach - could be enhanced with fuzzy matching
    similar_groups = {}
    for val in value_counts.index:
        if pd.isna(val) or str(val).lower() in ['nan', 'none', 'null']:
            continue
        # Look for similar values (simple case-insensitive matching)
        for existing_val in similar_groups:
            if val.lower() == existing_val.lower() or val.lower().replace(' ', '') == existing_val.lower().replace(' ', ''):
                similar_groups[existing_val].append(val)
                break
        else:
            similar_groups[val] = [val]
    
    # Apply grouping if beneficial
    grouped_count = 0
    for main_val, similar_vals in similar_groups.items():
        if len(similar_vals) > 1:
            df[col_name] = df[col_name].replace(similar_vals, main_val)
            grouped_count += len(similar_vals) - 1
    
    if grouped_count > 0:
        operations.append({
            'operation': 'group_similar_categories',
            'description': f'Grouped {grouped_count} similar category values',
            'values_grouped': grouped_count
        })
    
    # Handle missing values - fill with mode
    null_count = df[col_name].isnull().sum()
    if null_count > 0 and null_count < len(df) * 0.1:  # Less than 10% missing
        mode_value = df[col_name].mode()
        if not mode_value.empty:
            df[col_name].fillna(mode_value.iloc[0], inplace=True)
            operations.append({
                'operation': 'fill_missing_mode',
                'description': f'Filled {null_count} missing values with mode',
                'values_filled': int(null_count)
            })
    
    return df, operations

def clean_string_column(df: pd.DataFrame, col_name: str, col_info: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    """Clean string columns"""
    operations = []
    
    # Basic string cleaning
    df[col_name] = df[col_name].astype(str).str.strip()
    
    # Remove obvious placeholder values
    placeholder_patterns = ['n/a', 'na', 'null', 'none', 'nan', '', ' ']
    df[col_name] = df[col_name].replace(placeholder_patterns, pd.NA)
    
    operations.append({
        'operation': 'clean_string_placeholders',
        'description': 'Removed placeholder values and trimmed whitespace'
    })
    
    # Handle missing values if reasonable amount
    null_count = df[col_name].isnull().sum()
    if null_count > 0 and null_count < len(df) * 0.05:  # Less than 5% missing
        df[col_name].fillna('Unknown', inplace=True)
        operations.append({
            'operation': 'fill_missing_unknown',
            'description': f'Filled {null_count} missing string values with "Unknown"',
            'values_filled': int(null_count)
        })
    
    return df, operations

def generate_cleaned_data_preview(df: pd.DataFrame, cleaning_report: Dict, num_rows: int = 10) -> Dict[str, Any]:
    """Generate a preview of the cleaned dataset"""
    preview = {
        'cleaned_data': {
            'sample_rows': [],
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns)
        },
        'cleaning_summary': {
            'total_operations': len(cleaning_report.get('operations', [])),
            'rows_removed': cleaning_report.get('rows_removed', 0),
            'operations_by_type': {},
            'key_changes': []
        },
        'generated_at': datetime.utcnow().isoformat()
    }
    
    # Sample rows (convert to JSON-serializable format)
    sample_df = df.head(num_rows)
    for idx, row in sample_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            # Handle different data types for JSON serialization
            if pd.isna(value):
                row_dict[col] = None
            elif isinstance(value, (np.integer, np.floating)):
                row_dict[col] = float(value) if np.isfinite(value) else None
            elif isinstance(value, (pd.Timestamp)):
                row_dict[col] = value.isoformat()
            else:
                row_dict[col] = str(value)
        preview['cleaned_data']['sample_rows'].append(row_dict)
    
    # Summarize cleaning operations
    operations_count = {}
    key_changes = []
    
    for op in cleaning_report.get('operations', []):
        op_type = op.get('operation', 'unknown')
        operations_count[op_type] = operations_count.get(op_type, 0) + 1
        
        # Add key changes for display
        if op.get('values_filled', 0) > 0 or op.get('outliers_capped', 0) > 0 or op.get('values_grouped', 0) > 0:
            key_changes.append({
                'column': op.get('column', 'unknown'),
                'operation': op_type,
                'description': op.get('description', ''),
                'impact': op.get('values_filled', op.get('outliers_capped', op.get('values_grouped', 0)))
            })
    
    preview['cleaning_summary']['operations_by_type'] = operations_count
    preview['cleaning_summary']['key_changes'] = key_changes[:10]  # Top 10 changes
    
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