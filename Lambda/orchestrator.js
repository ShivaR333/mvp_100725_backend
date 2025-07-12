const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = require('uuid');
const { OpenAI } = require('openai');
const { LLMChain } = require('langchain/chains');
const { PromptTemplate } = require('langchain/prompts');

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
        
        // Extract and validate JWT token from Authorization header (like validate.js)
        const authHeader = event.headers?.Authorization || event.headers?.authorization;
        
        if (!authHeader) {
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
                    message: 'Authorization header is required'
                })
            };
        }
        
        // Extract and decode JWT token
        const token = authHeader.replace('Bearer ', '');
        let userId;
        
        try {
            // Simple JWT decode (payload is the middle part of the token)
            const parts = token.split('.');
            if (parts.length !== 3) {
                throw new Error('Invalid JWT format');
            }
            
            // Decode the payload (middle part)
            const payload = JSON.parse(Buffer.from(parts[1], 'base64').toString());
            console.log('Decoded JWT payload:', payload);
            
            // Extract user ID from JWT payload
            userId = payload.sub;
            if (!userId) {
                throw new Error('No user ID found in JWT');
            }
            
        } catch (decodeError) {
            console.error('JWT decode error:', decodeError);
            
            // Fallback to mock data if JWT decode fails (like validate.js)
            if (token === 'demo-token') {
                userId = 'demo_user';
            } else if (token === 'admin-token') {
                userId = 'admin';
            } else {
                return {
                    statusCode: 401,
                    headers: {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        error: 'Invalid token',
                        message: 'Unable to decode or validate the provided token'
                    })
                };
            }
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
        const conversationHistory = session.history
            .map(turn => `${turn.speaker}: ${turn.text}`)
            .join('\n');
        
        const systemPrompt = `You are an AI assistant helping users with causal analysis and data insights. 
Based on the conversation history, ask the next most relevant question to help the user with their analysis.
Keep questions focused, specific, and actionable. Avoid repeating questions already asked.

Conversation so far:
${conversationHistory}

Provide only your next question, nothing else.`;
        
        // Create LLM chain
        const model = new OpenAI({
            openAIApiKey: process.env.OPENAI_API_KEY,
            modelName: 'gpt-3.5-turbo',
            maxTokens: 150,
            temperature: 0.7
        });
        
        const prompt = PromptTemplate.fromTemplate(systemPrompt);
        const chain = new LLMChain({ llm: model, prompt });
        
        // Get LLM response
        let assistantMessage;
        try {
            const response = await chain.call({});
            assistantMessage = response?.text?.trim() || 'What would you like to analyze next?';
        } catch (error) {
            console.error('Error calling LLM chain:', error);
            // Fallback response
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
            statusCode: 600,
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
