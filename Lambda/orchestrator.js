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
        
        // Build conversation context
        const conversationContext = session.history
            .map(turn => `${turn.speaker}: ${turn.text}`)
            .join('\n');
        
        // Create comprehensive system prompt for causal analysis assistant
        const systemPrompt = `You are an expert causal analysis consultant for OptiFlux. Communicate like a management consultant - precise, concise, and actionable.

COMMUNICATION STYLE:
- Lead with the key insight or direct answer
- Be conversational but professional
- Use bullet points only when listing multiple items or recommendations
- Keep responses focused and specific
- Avoid lengthy explanations unless requested

EXPERTISE:
- Causal inference (Pearl's hierarchy, potential outcomes)
- Econometrics (IV, RDD, DID, matching, synthetic controls)  
- Causal discovery (PC, FCI, GES, NOTEARS)
- Business applications of causal analysis

RESPONSE GUIDELINES:
- Answer the question directly first
- Use bullets only for lists of multiple items, methods, or steps
- For single concepts or explanations, use natural sentences
- End with specific next steps when appropriate

Context: ${conversationContext}
Question: ${userMessage}

Provide a helpful, focused response. No thinking process or verbose explanations.`;

        // Call Cerebras API (OpenAI-compatible endpoint)
        let assistantMessage;
        try {
            console.log('Calling Cerebras API (Qwen-3-32B) for user message:', userMessage.substring(0, 100));
            
            const response = await fetch('https://api.cerebras.ai/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${process.env.CEREBRAS_API_KEY}`,
                    'Content-Type': 'application/json',
                    'User-Agent': 'OptiFlux/1.0' // Required to avoid CloudFront blocking
                },
                body: JSON.stringify({
                    model: 'qwen-3-32b',
                    messages: [
                        { role: 'system', content: systemPrompt },
                        { role: 'user', content: userMessage }
                    ],
                    max_completion_tokens: 500,
                    temperature: 0.7,
                    top_p: 1,
                    stream: false
                })
            });

            if (!response.ok) {
                const errorData = await response.text();
                throw new Error(`Cerebras API error: ${response.status} - ${errorData}`);
            }

            const data = await response.json();
            console.log('Cerebras API response received, choices count:', data.choices?.length);
            
            // Extract response and filter out thinking tokens (content between <thinking> tags)
            let rawResponse = data.choices[0]?.message?.content?.trim() || 'I apologize, but I wasn\'t able to generate a response. Could you please rephrase your question?';
            
            // Remove thinking tokens if present (both <thinking> and <think> variants)
            assistantMessage = rawResponse.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<thinking>[\s\S]*?<\/thinking>/gi, '').trim();
            
            // Fallback if response becomes empty after filtering
            if (!assistantMessage) {
                assistantMessage = 'I apologize, but I wasn\'t able to generate a response. Could you please rephrase your question?';
            }
            
        } catch (error) {
            console.error('Error calling Cerebras API:', {
                error: error.message,
                stack: error.stack,
                apiKey: process.env.CEREBRAS_API_KEY ? 'SET' : 'NOT_SET',
                userMessage: userMessage.substring(0, 100)
            });
            
            // More informative fallback that includes the actual error
            if (userMessage.toLowerCase().includes('causal')) {
                assistantMessage = 'I\'d be happy to help with causal analysis! Could you tell me more about your specific research question or dataset? For example, are you trying to estimate a treatment effect, identify causal relationships, or validate existing assumptions?';
            } else if (userMessage.toLowerCase().includes('data')) {
                assistantMessage = 'I can help you analyze your data for causal relationships. What type of data do you have (observational, experimental, time series) and what outcome are you trying to understand?';
            } else {
                assistantMessage = `I'm experiencing a technical issue with the AI service, but I'm here to help with causal analysis! Feel free to ask me about causal inference methods, econometric techniques, or your specific analytical challenge. (Debug: ${error.message})`;
            }
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
