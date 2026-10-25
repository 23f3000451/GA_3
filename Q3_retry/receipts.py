
import base64

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from schemas import PROFILE, canonical_bytes, proposal_digest


def verify_receipt_signature(jwk, evaluation_id, input_digest, receipt):
    try:
        if set(jwk) != {"kty", "crv", "x"}:
            return False
        if jwk["kty"] != "OKP" or jwk["crv"] != "Ed25519":
            return False

        encoded = jwk["x"]
        raw_key = base64.urlsafe_b64decode(
            encoded + "=" * (-len(encoded) % 4)
        )
        if len(raw_key) != 32:
            return False

        signature = base64.b64decode(
            receipt["receiptSignature"], validate=True
        )

        unsigned_receipt = {
            k: v for k, v in receipt.items() if k != "receiptSignature"
        }

        signed_object = {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "inputDigest": input_digest,
            "receipt": unsigned_receipt,
        }

        Ed25519PublicKey.from_public_bytes(raw_key).verify(
            signature, canonical_bytes(signed_object)
        )
        return True

    except (ValueError, TypeError, KeyError, InvalidSignature):
        return False
