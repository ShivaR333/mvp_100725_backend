exports.handler = async (event) => {
    try {
        console.log('Received event:', JSON.stringify(event, null, 2));
        
        // Check if this is an authorizer request
        if (event.type === 'REQUEST') {
            return handleAuthorizerRequest(event);
        }
        
        // Parse the event to get request details (API Gateway v2 format)
        const httpMethod = event.requestContext?.http?.method || event.httpMethod;
        const headers = event.headers || {};
        const queryStringParameters = event.queryStringParameters || {};
        const body = event.body;
        
        // Log the incoming request for debugging
        console.log('Received request:', { httpMethod, headers, queryStringParameters, fullEvent: JSON.stringify(event) });
        
        // Handle CORS preflight requests
        if (httpMethod === 'OPTIONS') {
            return {
                statusCode: 200,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                },
                body: JSON.stringify({ message: 'CORS preflight handled' })
            };
        }
        
        // Only allow GET requests for validation
        if (httpMethod !== 'GET') {
            return {
                statusCode: 405,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    error: 'Method not allowed',
                    message: 'Only GET requests are supported for validation' 
                })
            };
        }
        
        // Extract authorization header
        const authHeader = headers.Authorization || headers.authorization;
        
        if (!authHeader) {
            return {
                statusCode: 401,
                headers: {
                    'Access-Control-Allow-Origin': '*',
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
        
        // Decode JWT token (without verification for demo purposes)
        // In production, you should verify the token with WorkOS public key
        let userData;
        
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
            const userId = payload.sub;
            if (!userId) {
                throw new Error('No user ID found in JWT');
            }
            
            // Call WorkOS API to get user details
            // Note: In production, you would store the WorkOS API key in environment variables
            const workosApiKey = process.env.WORKOS_API_KEY; // You'll need to set this in Lambda environment
            
            if (workosApiKey) {
                try {
                    console.log('Fetching user details from WorkOS API for user:', userId);
                    
                    const response = await fetch(`https://api.workos.com/user_management/users/${userId}`, {
                        headers: {
                            'Authorization': `Bearer ${workosApiKey}`,
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const userDetails = await response.json();
                        console.log('WorkOS user details:', userDetails);
                        
                        userData = {
                            username: userDetails.first_name && userDetails.last_name 
                                ? `${userDetails.first_name} ${userDetails.last_name}` 
                                : userDetails.email || userId,
                            email: userDetails.email || `${userId}@workos.local`,
                            organization: userDetails.organization_name || 'WorkOS Organization'
                        };
                    } else {
                        console.error('Failed to fetch user details from WorkOS API:', response.status);
                        throw new Error('Failed to fetch user details');
                    }
                } catch (apiError) {
                    console.error('WorkOS API error:', apiError);
                    // Fall back to basic user data
                    userData = {
                        username: userId,
                        email: `${userId}@workos.local`,
                        organization: 'WorkOS Organization'
                    };
                }
            } else {
                // No API key available, use basic user data
                userData = {
                    username: userId,
                    email: `${userId}@workos.local`,
                    organization: 'WorkOS Organization'
                };
            }
            
        } catch (decodeError) {
            console.error('JWT decode error:', decodeError);
            
            // Fallback to mock data if JWT decode fails
            if (token === 'demo-token') {
                userData = {
                    username: 'demo_user',
                    email: 'demo@optiflux.com'
                };
            } else if (token === 'admin-token') {
                userData = {
                    username: 'admin',
                    email: 'admin@optiflux.com'
                };
            } else {
                return {
                    statusCode: 401,
                    headers: {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        error: 'Invalid token',
                        message: 'Unable to decode or validate the provided token' 
                    })
                };
            }
        }
        
        // Return successful validation response
        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                success: true,
                message: 'User validated successfully',
                ...userData
            })
        };
        
    } catch (error) {
        console.error('Error in validate handler:', error);
        
        return {
            statusCode: 500,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                error: 'Internal server error',
                message: 'An error occurred during validation'
            })
        };
    }
};

// Authorizer handler function
async function handleAuthorizerRequest(event) {
    try {
        const headers = event.headers || {};
        const authHeader = headers.Authorization || headers.authorization;
        
        if (!authHeader) {
            console.log('No authorization header found');
            return generatePolicy('user', 'Deny', event.routeArn);
        }
        
        // Extract and validate JWT token
        const token = authHeader.replace('Bearer ', '');
        
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
            const userId = payload.sub;
            if (!userId) {
                throw new Error('No user ID found in JWT');
            }
            
            // Return allow policy with user context
            return generatePolicy(userId, 'Allow', event.routeArn, {
                userId: userId,
                email: payload.email || `${userId}@workos.local`,
                // Pass additional claims that the orchestrator might need
                claims: JSON.stringify(payload)
            });
            
        } catch (decodeError) {
            console.error('JWT decode error:', decodeError);
            
            // Check for demo tokens
            if (token === 'demo-token' || token === 'admin-token') {
                return generatePolicy(token, 'Allow', event.routeArn, {
                    userId: token,
                    email: `${token}@optiflux.com`
                });
            }
            
            return generatePolicy('user', 'Deny', event.routeArn);
        }
        
    } catch (error) {
        console.error('Error in authorizer:', error);
        return generatePolicy('user', 'Deny', event.routeArn);
    }
}

// Generate IAM policy for API Gateway
function generatePolicy(principalId, effect, resource, context = {}) {
    const authResponse = {
        principalId: principalId,
        policyDocument: {
            Version: '2012-10-17',
            Statement: [
                {
                    Action: 'execute-api:Invoke',
                    Effect: effect,
                    Resource: resource
                }
            ]
        }
    };
    
    // Add context if provided (this will be available in the lambda as event.requestContext.authorizer)
    if (Object.keys(context).length > 0) {
        authResponse.context = context;
    }
    
    return authResponse;
}