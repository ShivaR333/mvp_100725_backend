const AWS = require('aws-sdk');

const dynamodb = new AWS.DynamoDB.DocumentClient();

const DYNAMODB_TABLE = process.env.DYNAMODB_TABLE;

exports.handler = async (event) => {
    console.log('Get dataset status triggered:', JSON.stringify(event, null, 2));
    
    try {
        // Extract dataset_id from path parameters
        const datasetId = event.pathParameters?.dataset_id;
        
        if (!datasetId) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                },
                body: JSON.stringify({
                    error: 'dataset_id is required'
                })
            };
        }
        
        // Extract user ID from JWT claims
        const userId = event.requestContext.authorizer?.claims?.sub || 'anonymous';
        
        console.log(`Getting status for dataset: ${datasetId}, user: ${userId}`);
        
        // Query DynamoDB for dataset metadata
        const result = await dynamodb.get({
            TableName: DYNAMODB_TABLE,
            Key: {
                user_id: userId,
                dataset_id: datasetId
            }
        }).promise();
        
        if (!result.Item) {
            return {
                statusCode: 404,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                },
                body: JSON.stringify({
                    error: 'Dataset not found'
                })
            };
        }
        
        const dataset = result.Item;
        
        // Prepare response with relevant information
        const response = {
            dataset_id: dataset.dataset_id,
            name: dataset.name,
            description: dataset.description || '',
            status: dataset.status,
            processing_message: dataset.processing_message || '',
            processing_step: dataset.processing_step || '',
            created_at: dataset.created_at,
            updated_at: dataset.updated_at,
            original_row_count: dataset.original_row_count || 0,
            final_row_count: dataset.final_row_count || 0
        };
        
        // Add completion details if processing is done
        if (dataset.status === 'completed') {
            response.processing_completed_at = dataset.processing_completed_at;
            response.cleaned_s3_key = dataset.cleaned_s3_key;
            
            if (dataset.schema_summary) {
                response.schema_summary = dataset.schema_summary;
            }
            
            if (dataset.cleaning_summary) {
                response.cleaning_summary = dataset.cleaning_summary;
            }
        }
        
        // Add error details if processing failed
        if (dataset.status === 'failed' && dataset.error_details) {
            response.error_details = {
                error_message: dataset.error_details.error_message,
                failed_at: dataset.error_details.failed_at
            };
        }
        
        // Add progress information for in-progress datasets
        if (dataset.status === 'processing') {
            const progressSteps = ['uploading', 'processing', 'schema_mapped', 'data_cleaned', 'completed'];
            const currentStepIndex = progressSteps.indexOf(dataset.processing_step || 'processing');
            response.progress = {
                current_step: dataset.processing_step || 'processing',
                progress_percentage: currentStepIndex >= 0 ? Math.round((currentStepIndex / (progressSteps.length - 1)) * 100) : 25
            };
        }
        
        console.log(`Status retrieved for dataset ${datasetId}: ${dataset.status}`);
        
        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            body: JSON.stringify(response)
        };
        
    } catch (error) {
        console.error('Error getting dataset status:', error);
        
        return {
            statusCode: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            body: JSON.stringify({
                error: 'Internal server error',
                message: error.message
            })
        };
    }
};