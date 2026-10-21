"""Optional pinned Ed25519 sender authentication for DataChannel blobs.

WebRTC DTLS protects transport but does not independently establish a human peer
identity in the existing anonymous signaling design. Pin a peer public key.
"""
from __future__ import annotations
import base64
import os
import struct
from pathlib import Path
from .core import PatchError

DOMAIN = b'Fedpara/WebRTC-artifact/v1\x00'


def _message(size: int, sha256: str) -> bytes:
    if not isinstance(size, int) or size < 1 or not isinstance(sha256, str) or len(sha256) != 64:
        raise PatchError('invalid signed transfer size/hash')
    try:
        return DOMAIN + struct.pack('!Q', size) + bytes.fromhex(sha256)
    except (ValueError, OverflowError) as exc:
        raise PatchError('invalid signed transfer') from exc


def generate_keypair(private: Path, public: Path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    private = Path(private); public = Path(public)
    if private.exists() or public.exists():
        raise FileExistsError('identity key already exists; refusing overwrite')
    sk = Ed25519PrivateKey.generate()
    pem = sk.private_bytes(serialization.Encoding.PEM,
                           serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd,'wb') as f: f.write(pem)
    pub = sk.public_key().public_bytes(serialization.Encoding.PEM,
                                      serialization.PublicFormat.SubjectPublicKeyInfo)
    fd = os.open(public, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd,'wb') as f: f.write(pub)


def signer_from_pem(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = serialization.load_pem_private_key(Path(path).read_bytes(),password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise PatchError('not an Ed25519 private key')
    def sign(size, digest):
        return base64.b64encode(key.sign(_message(size,digest))).decode('ascii')
    return sign


def verifier_from_pem(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise PatchError('not an Ed25519 public key')
    def verify(msg):
        signature = msg.get('sig')
        if not isinstance(signature,str) or len(signature) > 128:
            raise PatchError('unsigned transfer: pinned peer key required')
        try:
            raw = base64.b64decode(signature, validate=True)
            key.verify(raw, _message(msg['size'],msg['sha256']))
        except (ValueError, TypeError, KeyError, InvalidSignature) as exc:
            # Error intentionally generic, no untrusted identity is shown.
            raise PatchError('remote sender signature invalid') from exc
    return verify
