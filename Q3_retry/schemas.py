
import hashlib
import json
import re
from typing import Any

PROFILE = "ga5-mailroom-action-gate/v2"

CALL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")

# Tool arguments must be identifiers, not prose or copied secrets.
IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,255}$")
IDENT_FIELDS = {
    "referenceId", "sourceEventId", "artifactId",
    "claimedSender", "recipient",
}
ALLOWED_ACTIONS = [
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
]

PAYLOAD_KEYS = {
    "create_draft": {
        "recipient",
        "referenceId",
        "status",
        "template",
    },
    "update_internal_record": {
        "field",
        "sourceEventId",
        "value",
    },
    "send_approved_notice": {
        "referenceId",
        "status",
        "template",
    },
    "request_confirmation": {
        "claimedSender",
        "questionCode",
        "referenceId",
    },
    "quarantine_item": {
        "artifactId",
        "reasonCode",
    },
    "no_action": {
        "reasonCode",
        "referenceId",
    },
}

TARGET_KINDS = {
    "create_draft": "draft_queue",
    "update_internal_record": "case_record",
    "send_approved_notice": "email",
    "request_confirmation": "approval_queue",
    "quarantine_item": "security_queue",
    "no_action": None,
}

FIXED_PAYLOAD_VALUES = {
    "create_draft": {
        "template": "order_status",
    },
    "update_internal_record": {
        "field": "delivery_window",
    },
    "send_approved_notice": {
        "template": "approved_delivery_notice",
    },
    "request_confirmation": {
        "questionCode": "VERIFY_REQUEST",
    },
    "quarantine_item": {
        "reasonCode": "INDIRECT_PROMPT_INJECTION",
    },
}

NO_ACTION_REASONS = {
    "ALREADY_COMPLETED",
    "DUPLICATE",
    "INFORMATIONAL",
}


def canonical_bytes(value: Any) -> bytes:
    """Recursively sort JSON object keys and encode compact UTF-8 JSON."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return lowercase SHA-256 hex of canonical JSON."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def input_digest(dossiers: list[dict]) -> str:
    """Digest the dossiers array, preserving array order."""
    return sha256_json(dossiers)


def dossier_fingerprint(dossier: dict) -> str:
    """Fingerprint the complete dossier content."""
    return sha256_json(dossier)


def stable_call_id(dossier: dict) -> str:
    """Create a deterministic, dossier-content-based call ID."""
    return "call_" + dossier_fingerprint(dossier)[:40]


def proposal_digest(proposal: dict) -> str:
    """
    Hash exactly the normalized proposal fields required by the contract.
    Evidence strings are sorted before hashing.
    """
    normalized = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal.get("target"),
        "payload": proposal["payload"],
        "evidence": sorted(proposal["evidence"]),
    }
    return sha256_json(normalized)


def get_dossier_line_ids(dossier: dict) -> set[str]:
    """Collect line IDs available for evidence in this dossier."""
    result = set()

    for source in dossier.get("sources", []):
        for line in source.get("lines", []):
            line_id = line.get("lineId")
            if isinstance(line_id, str):
                result.add(line_id)

    return result


def validate_proposal(proposal: dict, dossier: dict) -> dict:
    """
    Validate proposal structure, allowed fields, target types,
    fixed contract values, and evidence references.

    This checks structural correctness, not whether the action is
    semantically authorized by the dossier.
    """
    if not isinstance(proposal, dict):
        raise ValueError("Proposal must be a JSON object")

    expected_keys = {
        "dossierId",
        "callId",
        "action",
        "target",
        "payload",
        "evidence",
    }

    if set(proposal) != expected_keys:
        raise ValueError("Invalid proposal keys")

    dossier_id = dossier.get("dossierId")

    if not isinstance(dossier_id, str) or not dossier_id:
        raise ValueError("Invalid dossier ID")

    if proposal["dossierId"] != dossier_id:
        raise ValueError("Dossier ID mismatch")

    call_id = proposal["callId"]

    if not isinstance(call_id, str) or not CALL_ID_RE.fullmatch(call_id):
        raise ValueError("Invalid call ID")

    action = proposal["action"]

    if action not in ALLOWED_ACTIONS:
        raise ValueError("Unsupported action")

    payload = proposal["payload"]

    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")

    if set(payload) != PAYLOAD_KEYS[action]:
        raise ValueError("Payload keys do not match action schema")

    for key, value in payload.items():
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Payload field {key!r} must be a nonempty string"
            )

    # Enforce action-specific constants from the contract.
    for key, expected_value in FIXED_PAYLOAD_VALUES.get(
        action, {}
    ).items():
        if payload[key] != expected_value:
            raise ValueError(
                f"Invalid {key!r} for action {action!r}"
            )

    if action == "no_action":
        if payload["reasonCode"] not in NO_ACTION_REASONS:
            raise ValueError("Invalid no_action reasonCode")

    # --- canary / raw-text guard -------------------------------------------
    # Arguments must look like identifiers. This blocks a model that copies
    # prose or a secret token from a dossier line into a tool argument.
    for field in IDENT_FIELDS:
        if field in payload:
            v = payload[field]
            if not IDENT_RE.fullmatch(v):
                raise ValueError(
                    f"Payload {field!r} is not an identifier: {v!r}"
                )
    # ------------------------------------------------------------------------

    target = proposal["target"]

    target = proposal["target"]
    expected_kind = TARGET_KINDS[action]

    if action == "no_action":
        if target is not None:
            raise ValueError("no_action target must be null")
    else:
        if not isinstance(target, dict):
            raise ValueError("Target must be an object")

        if set(target) != {"kind", "id"}:
            raise ValueError("Target must contain exactly kind and id")

        if target["kind"] != expected_kind:
            raise ValueError("Target kind does not match action")

        if not isinstance(target["id"], str) or not target["id"]:
            raise ValueError("Target ID must be a nonempty string")

        if action == "quarantine_item" and target["id"] != "mailroom":
            raise ValueError("quarantine_item target ID must be mailroom")

        # Target id shape checks
        if action == "create_draft":
            if not target["id"].startswith("mailbox:"):
                raise ValueError("draft_queue target id must start with mailbox:")
            addr = target["id"][len("mailbox:"):]
            if not IDENT_RE.fullmatch(addr) or "@" not in addr:
                raise ValueError("draft_queue target id has invalid mailbox")
        elif action == "send_approved_notice":
            if not IDENT_RE.fullmatch(target["id"]) or "@" not in target["id"]:
                raise ValueError("email target id must be a plain address")
        else:
            # case_record, approval_queue, security_queue ids
            if not IDENT_RE.fullmatch(target["id"]):
                raise ValueError("target id is not an identifier")

    evidence = proposal["evidence"]

    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Evidence must be a nonempty array")

    if any(not isinstance(item, str) or not item for item in evidence):
        raise ValueError("Evidence IDs must be nonempty strings")

    if len(evidence) != len(set(evidence)):
        raise ValueError("Evidence IDs must be unique")

    available_ids = get_dossier_line_ids(dossier)

    if not set(evidence).issubset(available_ids):
        raise ValueError("Evidence contains unknown line IDs")

    return proposal


AUTHORITY_WORDS = ("approve", "approved", "authorize", "authorized",
                   "case", "sent", "completed", "duplicate", "ignore previous")

def trim_evidence(proposal, dossier):
    """Keep the smallest set of lines that prove authority or arguments."""
    if len(proposal["evidence"]) <= 2:
        return proposal
    # map lineId -> text
    texts = {}
    for src in dossier.get("sources", []):
        for ln in src.get("lines", []):
            texts[ln["lineId"]] = ln["text"].lower()
    # collect argument values to look for
    needles = [str(v).lower() for v in proposal["payload"].values()
               if isinstance(v, str) and len(v) >= 3]
    if proposal.get("target"):
        needles.append(str(proposal["target"]["id"]).lower())
    keep = []
    for lid in proposal["evidence"]:
        t = texts.get(lid, "")
        if any(n in t for n in needles) or any(w in t for w in AUTHORITY_WORDS):
            keep.append(lid)
    # always keep at least the first cited line
    if not keep:
        keep = [proposal["evidence"][0]]
    proposal["evidence"] = keep[:3]
    return proposal
