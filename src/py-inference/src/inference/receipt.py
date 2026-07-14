"""Signed inference receipts (design-improvement A1).

The worker attests, per served request: WHAT it was asked (sha256 of the exact request
body it received), WHAT it produced (sha256 of the generated text), and the token counts
it reports for billing — signed with a worker-held ed25519 key whose pubkey is advertised
on /health. The gateway verifies and stores the receipt in the billing ledger; settlement
commits a Merkle root over receipts into the on-chain batch hash. Result: a client (or an
SP) can verify offline that the charge settled on-chain matches what the worker actually
attested — the operator can no longer silently fabricate usage.

Signing payload: a FIXED-TEMPLATE canonical JSON string (field order hardcoded, string
values individually JSON-encoded). Both this module and the gateway's Go verifier build
the exact same bytes — no reliance on any JSON library's key ordering.
"""
import json
import logging
import os
import time

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

_signer: "ReceiptSigner | None" = None


class ReceiptSigner:
    def __init__(self, key: ed25519.Ed25519PrivateKey, persistent: bool):
        self._key = key
        self.pubkey_hex = key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        self.persistent = persistent

    def sign(self, payload: bytes) -> str:
        return self._key.sign(payload).hex()


def get_signer() -> ReceiptSigner:
    """Load (or create) the worker's receipt signing key. Falls back to an ephemeral
    in-memory key if the key path is unwritable — receipts stay functional, but the
    pubkey changes on restart (the gateway logs pubkey changes)."""
    global _signer
    if _signer is not None:
        return _signer
    # Key location: on the shared /models volume by default so it survives container
    # rebuilds (same lifecycle as the model cache). Override with WORKER_RECEIPT_KEY.
    path = os.environ.get("WORKER_RECEIPT_KEY", "/models/.openmodel/receipt-ed25519.key")
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                key = ed25519.Ed25519PrivateKey.from_private_bytes(f.read())
            _signer = ReceiptSigner(key, persistent=True)
            logger.info("receipt signing key loaded (pubkey=%s…)", _signer.pubkey_hex[:16])
            return _signer
        key = ed25519.Ed25519PrivateKey.generate()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw = key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        _signer = ReceiptSigner(key, persistent=True)
        logger.info("receipt signing key generated (pubkey=%s…)", _signer.pubkey_hex[:16])
    except OSError as e:
        logger.warning("receipt key path %s unusable (%s) — using EPHEMERAL key "
                       "(pubkey changes on restart)", path, e)
        _signer = ReceiptSigner(ed25519.Ed25519PrivateKey.generate(), persistent=False)
    return _signer


def canonical_payload(request_id: str, model: str, request_sha256: str,
                      response_sha256: str, prompt_tokens: int, completion_tokens: int,
                      cached_tokens: int, ts: int, pubkey: str) -> bytes:
    """Fixed-template canonical bytes — MUST stay byte-identical with the Go verifier
    (sp-state-agent internal/gateway/receipt.go canonicalReceiptPayload)."""
    return ("{" +
            f'"cached_tokens":{cached_tokens},'
            f'"completion_tokens":{completion_tokens},'
            f'"model":{json.dumps(model)},'
            f'"prompt_tokens":{prompt_tokens},'
            f'"pubkey":{json.dumps(pubkey)},'
            f'"request_id":{json.dumps(request_id)},'
            f'"request_sha256":{json.dumps(request_sha256)},'
            f'"response_sha256":{json.dumps(response_sha256)},'
            f'"ts":{ts},'
            f'"v":1' +
            "}").encode("utf-8")


def build_receipt(request_id: str, model: str, request_sha256: str, response_sha256: str,
                  prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict:
    """Build and sign a receipt dict (wire form, sent to the gateway)."""
    s = get_signer()
    ts = int(time.time())
    payload = canonical_payload(request_id, model, request_sha256, response_sha256,
                                prompt_tokens, completion_tokens, cached_tokens,
                                ts, s.pubkey_hex)
    return {
        "v": 1,
        "request_id": request_id,
        "model": model,
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "ts": ts,
        "pubkey": s.pubkey_hex,
        "sig": s.sign(payload),
    }
