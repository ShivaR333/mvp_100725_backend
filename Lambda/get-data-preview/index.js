const AWS = require('aws-sdk');

const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();

const S3_BUCKET = process.env.S3_BUCKET_NAME;
const DYNAMODB_TABLE = process.env.DYNAMODB_TABLE;

exports.handler = async (event) => {
    console.log('Data preview request:', JSON.stringify(event, null, 2));
    
    try {
        // Extract parameters from API Gateway event
        const { dataset_id } = event.pathParameters || {};
        const { preview_type = 'all' } = event.queryStringParameters || {};
        
        // Get user ID from headers (assuming authentication middleware sets this)
        const user_id = event.headers['x-user-id'] || 'anonymous';
        
        if (!dataset_id) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: JSON.stringify({ 
                    error: 'dataset_id is required' 
                })
            };
        }
        
        console.log(`Getting data preview for dataset: ${dataset_id}, user: ${user_id}, type: ${preview_type}`);
        
        // Check if dataset exists and belongs to user
        const datasetResponse = await dynamodb.get({
            TableName: DYNAMODB_TABLE,
            Key: { user_id, dataset_id }
        }).promise();
        
        if (!datasetResponse.Item) {
            return {
                statusCode: 404,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: JSON.stringify({ 
                    error: 'Dataset not found' 
                })
            };
        }
        
        const dataset = datasetResponse.Item;
        const baseKey = `datasets/${user_id}/${dataset_id}`;
        
        // Collect all preview data
        const previewData = {
            dataset_info: {
                dataset_id,
                status: dataset.status,
                created_at: dataset.created_at,
                updated_at: dataset.updated_at,
                processing_step: dataset.processing_step
            }
        };
        
        // Get different preview types based on request
        if (preview_type === 'raw' || preview_type === 'all') {
            try {
                const rawPreview = await getS3Object(S3_BUCKET, `${baseKey}/preview.json`);
                previewData.raw_preview = JSON.parse(rawPreview);
            } catch (err) {
                console.log('Raw preview not available:', err.message);
                previewData.raw_preview = null;
            }
        }
        
        if (preview_type === 'cleaned' || preview_type === 'all') {
            try {
                const cleanedPreview = await getS3Object(S3_BUCKET, `${baseKey}/cleaned_preview.json`);
                previewData.cleaned_preview = JSON.parse(cleanedPreview);
            } catch (err) {
                console.log('Cleaned preview not available:', err.message);
                previewData.cleaned_preview = null;
            }
        }
        
        if (preview_type === 'schema' || preview_type === 'all') {
            try {
                const schema = await getS3Object(S3_BUCKET, `${baseKey}/schema.json`);
                previewData.schema = JSON.parse(schema);
            } catch (err) {
                console.log('Schema not available:', err.message);
                previewData.schema = null;
            }
        }
        
        if (preview_type === 'cleaning_report' || preview_type === 'all') {
            try {
                const cleaningReport = await getS3Object(S3_BUCKET, `${baseKey}/cleaning_report.json`);
                previewData.cleaning_report = JSON.parse(cleaningReport);
            } catch (err) {
                console.log('Cleaning report not available:', err.message);
                previewData.cleaning_report = null;
            }
        }
        
        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify(previewData)
        };
        
    } catch (error) {
        console.error('Error getting data preview:', error);
        
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

async function getS3Object(bucket, key) {
    const response = await s3.getObject({
        Bucket: bucket,
        Key: key
    }).promise();
    
    return response.Body.toString('utf-8');
}