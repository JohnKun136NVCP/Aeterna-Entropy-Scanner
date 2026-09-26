import secrets
import os


def generateTokenCiphers(cipher: str):
    if cipher == "DES":
        # Full 0-255 byte range, same quality as RC4/AES below.
        # (Previously restricted to ASCII letters+digits, which fixed the
        # high bit of every key byte to 0 -- a real, structural bias.)
        return secrets.token_bytes(8)
    elif cipher == "RC4":
        return secrets.token_bytes(16)
    elif cipher == "AES":
        return secrets.token_bytes(32)
    else:
        return 0


def generate_iv(cipher: str) -> bytes:
    if cipher == "DES":
        return os.urandom(8)
    else:
        return os.urandom(16)