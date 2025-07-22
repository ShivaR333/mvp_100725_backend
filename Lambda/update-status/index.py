import json
import boto3
import os
from datetime import datetime

dynamodb = boto3.resource('dynamodb')

DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
table = dynamodb.Table(DYNAMODB_TABLE)

def lambda_handler(event, context):
    """
    Lambda function to update dataset status to failed when errors occur
    """
    print(f"Update status triggered: {json.dumps(event, default=str)}")
    
    try:
        # Extract error information from Step Functions error
        error_info = event.get('Error', 'Unknown error')
        cause = event.get('Cause', '')
        
        # Try to extract dataset info from the original event
        # This might be in different places depending on where the error occurred
        user_id = None
        dataset_id = None
        
        # Look for dataset info in various places
        if 'user_id' in event and 'dataset_id' in event:
            user_id = event['user_id']
            dataset_id = event['dataset_id']
        elif 'Input' in event:
            input_data = json.loads(event['Input']) if isinstance(event['Input'], str) else event['Input']
            user_id = input_data.get('user_id')
            dataset_id = input_data.get('dataset_id')
        
        if not user_id or not dataset_id:
            print("Warning: Could not extract user_id or dataset_id from error event")
            return {
                'status': 'error',
                'message': 'Could not identify dataset for error update'
            }
        
        # Parse the cause to get more detailed error information
        error_details = error_info
        if cause:
            try:
                cause_data = json.loads(cause)
                if 'errorMessage' in cause_data:
                    error_details = cause_data['errorMessage']
            except:
                error_details = cause
        
        print(f"Updating failed status for dataset: {dataset_id}, error: {error_details}")
        
        # Update DynamoDB record with failed status
        now = datetime.utcnow().isoformat()
        
        table.update_item(
            Key={'user_id': user_id, 'dataset_id': dataset_id},
            UpdateExpression='''SET 
                #status = :status,
                updated_at = :updated,
                processing_message = :message,
                error_details = :error,
                failed_at = :failed_at
            ''',
            ExpressionAttributeValues={
                ':status': 'failed',
                ':updated': now,
                ':message': f'Processing failed: {error_details}',
                ':error': {
                    'error_type': error_info,
                    'error_message': error_details,
                    'failed_at': now
                },
                ':failed_at': now
            },
            ExpressionAttributeNames={
                '#status': 'status'
            }
        )
        
        print(f"Successfully updated failed status for dataset: {dataset_id}")
        
        return {
            'status': 'failed',
            'dataset_id': dataset_id,
            'user_id': user_id,
            'error_details': error_details,
            'updated_at': now
        }
        
    except Exception as e:
        print(f"Error in update status function: {str(e)}")
        # Even if updating status fails, we don't want to throw another error
        return {
            'status': 'error',
            'message': f'Failed to update error status: {str(e)}'
        }