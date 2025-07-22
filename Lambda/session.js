const { WorkOS } = require('@workos-inc/node');

// Initialize WorkOS client
const workos = new WorkOS(process.env.WORKOS_API_KEY, {
    clientId: process.env.WORKOS_CLIENT_ID,
});

/**
 * Session Lambda Function - Handles WorkOS code exchange for sealed sessions
 * 
 * This function:
 * 1. Receives authorization code from WorkOS AuthKit
 * 2. Exchanges code for user info and sealed session
 * 3. Sets secure session cookie
 * 4. Returns user information to frontend
 */
exports.handler = async (event) => {
    try {
        console.log('Session Lambda - Received event:', JSON.stringify(event, null, 2));
        
        // Handle CORS preflight requests
        const httpMethod = event.requestContext?.http?.method || event.httpMethod;
        if (httpMethod === 'OPTIONS') {
            return {
                statusCode: 200,
                headers: {
                    'Access-Control-Allow-Origin': 'http://localhost:3000',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Access-Control-Max-Age': '86400',
                },
                body: JSON.stringify({ message: 'CORS preflight handled' })
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
                    'Access-Control-Allow-Origin': 'http://localhost:3000',
                    'Access-Control-Allow-Credentials': 'true',
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
        
        const { code } = requestBody;
        
        // Validate authorization code
        if (!code || typeof code !== 'string') {
            return {
                statusCode: 400,
                headers: {
                    'Access-Control-Allow-Origin': 'http://localhost:3000',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Bad Request',
                    message: 'Authorization code is required'
                })
            };
        }
        
        // Exchange authorization code for sealed session
        let authenticateResponse;
        try {
            authenticateResponse = await workos.userManagement.authenticateWithCode({
                clientId: process.env.WORKOS_CLIENT_ID,
                code,
                session: {
                    sealSession: true,
                    cookiePassword: process.env.WORKOS_COOKIE_PASSWORD,
                },
            });
            
            console.log('WorkOS authentication successful for user:', authenticateResponse.user.email);
            
        } catch (error) {
            console.error('WorkOS authentication failed:', error);
            return {
                statusCode: 401,
                headers: {
                    'Access-Control-Allow-Origin': 'http://localhost:3000',
                    'Access-Control-Allow-Credentials': 'true',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: 'Authentication Failed',
                    message: 'Invalid or expired authorization code'
                })
            };
        }
        
        const { user, sealedSession } = authenticateResponse;
        
        console.log('Setting wos-session cookie for user:', user.email);
        console.log('Sealed session length:', sealedSession.length);
        
        // Prepare response headers with secure session cookie
        const responseHeaders = {
            'Access-Control-Allow-Origin': 'http://localhost:3000',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Cookie',
            'Access-Control-Allow-Credentials': 'true',
            'Content-Type': 'application/json',
            'Set-Cookie': `wos-session=${encodeURIComponent(sealedSession)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400`
        };
        
        // Return success response with user information
        return {
            statusCode: 200,
            headers: responseHeaders,
            body: JSON.stringify({
                success: true,
                user: {
                    id: user.id,
                    email: user.email,
                    firstName: user.firstName,
                    lastName: user.lastName,
                    profilePictureUrl: user.profilePictureUrl
                },
                message: 'Session created successfully'
            })
        };
        
    } catch (error) {
        console.error('Error in session handler:', error);
        
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
                message: 'An unexpected error occurred during session creation'
            })
        };
    }
};