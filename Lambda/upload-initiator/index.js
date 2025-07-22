const AWS = require('aws-sdk');
const { v4: uuidv4 } = require('uuid');

const s3 = new AWS.S3();
const dynamodb = new AWS.DynamoDB.DocumentClient();

const S3_BUCKET = process.env.S3_BUCKET_NAME;
const DYNAMODB_TABLE = process.env.DYNAMODB_TABLE;
const REGION = process.env.REGION;

exports.handler = async (event) => {
    console.log('Upload initiator triggered (Node.js 16.x with UUID):', JSON.stringify(event, null, 2));
    
    try {
        // Parse the request body
        const body = JSON.parse(event.body);
        const { dataset_name, description } = body;
        
        // Extract user ID from JWT claims (assuming it's in the context)
        const userId = event.requestContext.authorizer?.claims?.sub || 'anonymous';
        
        // Validate required fields
        if (!dataset_name) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                },
                body: JSON.stringify({
                    error: 'dataset_name is required'
                })
            };
        }
        
        // Generate unique dataset ID
        const datasetId = uuidv4();
        const s3Key = `datasets/${userId}/${datasetId}/raw.csv`;
        
        // Create presigned URL for S3 upload
        const presignedPost = s3.createPresignedPost({
            Bucket: S3_BUCKET,
            Fields: {
                key: s3Key,
                'Content-Type': 'text/csv',
                'x-amz-meta-dataset-id': datasetId,
                'x-amz-meta-user-id': userId,
                'x-amz-meta-dataset-name': dataset_name
            },
            Conditions: [
                ['content-length-range', 0, 100 * 1024 * 1024], // 100MB max
                ['eq', '$Content-Type', 'text/csv'],
                ['starts-with', '$key', `datasets/${userId}/`]
            ],
            Expires: 600 // 10 minutes
        });
        
        // Store initial metadata in DynamoDB
        const now = new Date();
        const datasetMetadata = {
            user_id: userId,
            dataset_id: datasetId,
            name: dataset_name,
            description: description || '',
            status: 'uploading',
            s3_key: s3Key,
            created_at: now.toISOString(),
            updated_at: now.toISOString()
        };
        
        await dynamodb.put({
            TableName: DYNAMODB_TABLE,
            Item: datasetMetadata
        }).promise();
        
        console.log('Dataset metadata stored:', datasetId);
        
        // Return presigned URL and dataset info
        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                presigned_url: presignedPost.url,
                fields: presignedPost.fields,
                expires_at: new Date(Date.now() + 600 * 1000).toISOString()
            })
        };
        
    } catch (error) {
        console.error('Error in upload initiator:', error);
        
        return {
            statusCode: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            body: JSON.stringify({
                error: 'Internal server error',
                message: error.message
            })
        };
    }
};