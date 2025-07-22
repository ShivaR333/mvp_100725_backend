import json
import boto3
import os
from datetime import datetime
from typing import Dict, Any

dynamodb = boto3.resource('dynamodb')
events_client = boto3.client('events')

DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
EVENT_BUS_NAME = os.environ['EVENT_BUS_NAME']
table = dynamodb.Table(DYNAMODB_TABLE)

def lambda_handler(event, context):
    """
    Lambda function to save final dataset metadata and trigger completion events
    """
    print(f"Save data triggered: {json.dumps(event, default=str)}")
    
    try:
        # Extract information from previous step
        user_id = event.get('user_id')
        dataset_id = event.get('dataset_id')
        cleaned_key = event.get('cleaned_key')
        report_key = event.get('report_key')
        schema_key = event.get('schema_key')
        cleaning_report = event.get('cleaning_report', {})
        schema_info = event.get('schema_info', {})
        original_row_count = event.get('original_row_count', 0)
        cleaned_row_count = event.get('cleaned_row_count', 0)
        
        if not all([user_id, dataset_id, cleaned_key]):
            raise ValueError("Missing required parameters from previous step")
        
        print(f"Saving final metadata for dataset: {dataset_id}")
        
        # Update status to completed
        now = datetime.utcnow().isoformat()
        
        # Prepare final metadata update
        update_expression = '''SET 
            #status = :status,
            cleaned_s3_key = :cleaned_key,
            cleaning_report_key = :report_key,
            schema_key = :schema_key,
            final_row_count = :final_rows,
            original_row_count = :orig_rows,
            processing_completed_at = :completed,
            updated_at = :updated,
            processing_message = :message,
            processing_step = :step
        '''
        
        expression_values = {
            ':status': 'completed',
            ':cleaned_key': cleaned_key,
            ':report_key': report_key,
            ':schema_key': schema_key,
            ':final_rows': cleaned_row_count,
            ':orig_rows': original_row_count,
            ':completed': now,
            ':updated': now,
            ':message': f'Processing completed. {original_row_count} rows processed, {cleaned_row_count} rows in final dataset.',
            ':step': 'completed'
        }
        
        expression_names = {
            '#status': 'status'
        }
        
        # Add cleaning summary if available
        if cleaning_report:
            update_expression += ', cleaning_summary = :cleaning_summary'
            expression_values[':cleaning_summary'] = {
                'operations_count': len(cleaning_report.get('operations', [])),
                'rows_removed': cleaning_report.get('rows_removed', 0),
                'columns_processed': cleaning_report.get('columns_processed', 0)
            }
        
        # Add schema summary if available
        if schema_info:
            update_expression += ', schema_summary = :schema_summary'
            expression_values[':schema_summary'] = {
                'column_count': schema_info.get('column_count', 0),
                'total_columns': len(schema_info.get('columns', [])),
                'memory_usage': schema_info.get('memory_usage', 0)
            }
        
        # Update DynamoDB record
        table.update_item(
            Key={'user_id': user_id, 'dataset_id': dataset_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames=expression_names
        )
        
        print(f"Dataset metadata updated successfully: {dataset_id}")
        
        # Send completion event to EventBridge
        try:
            completion_event = {
                'Source': 'optiflux.data-processing',
                'DetailType': 'Dataset Processing Completed',
                'Detail': json.dumps({
                    'user_id': user_id,
                    'dataset_id': dataset_id,
                    'completed_at': now,
                    'original_row_count': original_row_count,
                    'final_row_count': cleaned_row_count,
                    'cleaned_s3_key': cleaned_key,
                    'processing_duration_seconds': calculate_processing_duration(event)
                })
            }
            
            events_client.put_events(
                Entries=[completion_event]
            )
            
            print(f"Completion event sent for dataset: {dataset_id}")
            
        except Exception as e:
            print(f"Warning: Failed to send completion event: {str(e)}")
            # Don't fail the entire process for event sending issues
        
        # Return success response
        return {
            'status': 'completed',
            'dataset_id': dataset_id,
            'user_id': user_id,
            'cleaned_s3_key': cleaned_key,
            'final_row_count': cleaned_row_count,
            'original_row_count': original_row_count,
            'processing_completed_at': now
        }
        
    except Exception as e:
        print(f"Error in save data: {str(e)}")
        
        # Update status to failed
        try:
            if user_id and dataset_id:
                update_status(user_id, dataset_id, 'failed', f'Save data failed: {str(e)}')
        except Exception as inner_e:
            print(f"Failed to update error status: {str(inner_e)}")
        
        raise e

def calculate_processing_duration(event: Dict) -> int:
    """
    Calculate processing duration if timestamps are available
    """
    try:
        # This would need to be enhanced to track actual start time
        # For now, return a default or calculate from event timestamps
        return 0
    except:
        return 0

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