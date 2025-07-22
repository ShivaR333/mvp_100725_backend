const AWS = require('aws-sdk');

const dynamodb = new AWS.DynamoDB.DocumentClient();

const DYNAMODB_TABLE = process.env.DYNAMODB_TABLE;

exports.handler = async (event) => {
    console.log('List datasets request (Node.js 16.x with AWS SDK):', JSON.stringify(event, null, 2));
    
    try {
        // Get user ID from headers (assuming authentication middleware sets this)
        const user_id = event.headers['x-user-id'] || 'anonymous';
        
        console.log(`Getting all datasets for user: ${user_id}`);
        
        // Query DynamoDB for all datasets belonging to this user
        const response = await dynamodb.query({
            TableName: DYNAMODB_TABLE,
            KeyConditionExpression: 'user_id = :user_id',
            ExpressionAttributeValues: {
                ':user_id': user_id
            },
            // Sort by created_at in descending order (newest first)
            ScanIndexForward: false
        }).promise();
        
        const datasets = response.Items.map(item => ({
            dataset_id: item.dataset_id,
            name: item.dataset_name || item.name,
            status: item.status,
            created_at: item.created_at,
            updated_at: item.updated_at,
            processing_step: item.processing_step,
            processing_message: item.processing_message,
            original_row_count: item.original_row_count,
            final_row_count: item.final_row_count,
            schema_summary: item.schema_summary,
            cleaning_summary: item.cleaning_summary,
            description: item.description
        }));
        
        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify({
                datasets,
                count: datasets.length
            })
        };
        
    } catch (error) {
        console.error('Error listing datasets:', error);
        
        return {
            statusCode: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify({ 
                error: 'Internal server error',
                message: error.message 
            })
        };
    }
};