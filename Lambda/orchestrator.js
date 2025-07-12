const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = require('uuid');

const dynamoClient = new DynamoDBClient({ region: process.env.AWS_REGION || 'us-east-1' });
const docClient = DynamoDBDocumentClient.from(dynamoClient);

exports.handler = async (event) => {
    try {
        console.log('Received event:', JSON.stringify(event, null, 2));
        
        // Handle CORS preflight requests
        const httpMethod = event.requestContext?.http?.method || event.httpMethod;
        if (httpMethod === 'OPTIONS') {
            return {
                statusCode: 200,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Access-Control-Max-Age': '86400',
                },
                body: JSON.stringify({ message: 'CORS preflight handled' })
            };
        }
        
        // Extract user ID from the JWT authorizer context or fallback to header
        let userId = event.requestContext?.authorizer?.jwt?.claims?.sub;

        if (!userId) {
            const authHeader = event.headers?.Authorization || event.headers?.authorization;
            if (authHeader && authHeader.startsWith('Bearer ')) {
                const token = authHeader.replace('Bearer ', '');
                try {
                    const parts = token.split('.');
                    if (parts.length === 3) {
                        const payload = JSON.parse(Buffer.from(parts[1], 'base64').toString());
                        userId = payload.sub || 'guest';
                    }
                } catch (e) {
                    console.warn('Failed to decode JWT token for userId fallback:', e);
                }
            }
            // As a final fallback, use anonymous user
            if (!userId) userId = 'guest';
        }
        
        if (!userId) {
            return {
                statusCode: 401,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Unauthorized',
                    message: 'User ID not found in JWT token'
                })
            };
        }
        
        // Parse request body
        let requestBody;
        try {
            requestBody = JSON.parse(event.body || '{}');
        } catch (error) {
            return {
                statusCode: 400,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Bad Request',
                    message: 'Invalid JSON in request body'
                })
            };
        }
        
        const { sessionId: providedSessionId, message: userMessage } = requestBody;
        
        if (!userMessage || typeof userMessage !== 'string') {
            return {
                statusCode: 400,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Bad Request',
                    message: 'Message is required and must be a string'
                })
            };
        }
        
        // Generate session ID if not provided
        const sessionId = providedSessionId || uuidv4();
        const now = new Date().toISOString();
        
        // Load or create session
        let session;
        try {
            const getParams = {
                TableName: process.env.DYNAMODB_TABLE_NAME,
                Key: {
                    PK: `USER#${userId}`,
                    SK: `SESSION#${sessionId}`
                }
            };
            
            const getResult = await docClient.send(new GetCommand(getParams));
            session = getResult.Item;
            
            if (!session) {
                // Create new session
                session = {
                    PK: `USER#${userId}`,
                    SK: `SESSION#${sessionId}`,
                    history: [],
                    createdAt: now,
                    lastUpdatedAt: now,
                    ttl: Math.floor(Date.now() / 1000) + (30 * 24 * 60 * 60) // 30 days TTL
                };
            }
        } catch (error) {
            console.error('Error accessing DynamoDB:', error);
            return {
                statusCode: 503,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Internal Server Error',
                    message: 'Failed to access session data'
                })
            };
        }
        
        // Append user turn to history
        const userTurn = {
            turn: session.history.length + 1,
            speaker: 'user',
            text: userMessage,
            timestamp: now
        };
        session.history.push(userTurn);
        
        // Build prompt from history

        const systemPrompt = `You are an AI assistant helping users with causal analysis and data insights.
Your goal is to ask the next most relevant question that will move the analysis forward.
Be concise, specific, and avoid repeating questions already asked.`;

        // Convert session history to ChatGPT messages format
        const chatMessages = [
            {
                role: 'system',
                content: systemPrompt
            }
        ];

        for (const turn of session.history) {
            chatMessages.push({
                role: turn.speaker === 'assistant' ? 'assistant' : 'user',
                content: turn.text
            });
        }
 
        // Call LLM for next question
        let assistantMessage;
        try {
            const response = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'gpt-3.5-turbo',
                    messages: chatMessages,
                    max_tokens: 150,
                    temperature: 0.7
                })
            });
            
            if (!response.ok) {
                throw new Error(`OpenAI API error: ${response.status}`);
            }
            
            const data = await response.json();
            assistantMessage = data.choices[0]?.message?.content?.trim() || 'What would you like to analyze next?';
            
        } catch (error) {
            console.error('Error calling OpenAI API:', error);
            // Fallback to a generic question
            assistantMessage = 'What specific aspect of your data would you like to explore further?';
        }
        
        // Append assistant turn to history
        const assistantTurn = {
            turn: session.history.length + 1,
            speaker: 'assistant',
            text: assistantMessage,
            timestamp: new Date().toISOString()
        };
        session.history.push(assistantTurn);
        
        // Update session metadata
        session.lastUpdatedAt = new Date().toISOString();
        
        // Persist session to DynamoDB
        try {
            const putParams = {
                TableName: process.env.DYNAMODB_TABLE_NAME,
                Item: session
            };
            
            await docClient.send(new PutCommand(putParams));
        } catch (error) {
            console.error('Error saving session to DynamoDB:', error);
            return {
                statusCode: 501,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Internal Server Error',
                    message: 'Failed to save session data'
                })
            };
        }
        
        // Return response
        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sessionId: sessionId,
                message: assistantMessage,
                turn: assistantTurn.turn
            })
        };
        
    } catch (error) {
        console.error('Error in orchestrator handler:', error);
        
        return {
            statusCode: 500,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                error: 'Internal Server Error',
                message: 'An unexpected error occurred'
            })
        };
    }
};