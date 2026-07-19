
from api.utils.auth import create_access_token, verify_access_token
from jose import jwt, JWTError

try:
    verify_access_token(None)
except Exception as e:
    print('Error with None:', type(e))
    
try:
    verify_access_token('Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...')
except Exception as e:
    print('Error with Bearer string:', type(e))
