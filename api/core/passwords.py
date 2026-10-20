from passlib.context import CryptContext


# bcrypt_sha256 handles passwords longer than bcrypt's 72-byte input limit
# without truncating them. Keep bcrypt enabled to verify existing hashes.
pwd_context = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated=["bcrypt"],
)


def hash_password(password: str) -> str:
    """Hash a password without bcrypt's 72-byte truncation limit."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify current and legacy hashes, safely rejecting oversized legacy input."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except (TypeError, ValueError):
        # Do not truncate long input when checking an old bcrypt-only hash.
        return False
