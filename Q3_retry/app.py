
import hashlib
import json
import os
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from decision import make_decision
from receipts import verify_receipt_signature
from schemas import (
    PROFILE,
    canonical_bytes,
    dossier_fingerprint,
    input_digest,
    proposal_digest,
    stable_call_id,
    validate_proposal,
)

DB_PATH = os.getenv("DATABASE_PATH", "/var/data/mailroom.db")
MAX_BODY = 2 * 1024 * 1024

def _fallback_proposal(dossier):
    """Conservative proposal when the model fails.

    Uses the first lineId of the dossier as evidence so the schema stays valid.
    """
    line_ids = [
        line["lineId"]
        for source in dossier.get("sources", [])
        for line in source.get("lines", [])
        if isinstance(line.get("lineId"), str) and line["lineId"]
    ]
    if not line_ids:
        raise HTTPException(422, "Dossier has no usable evidence lines")
    return {
        "dossierId": dossier["dossierId"],
        "callId": stable_call_id(dossier),
        "action": "request_confirmation",
        "target": {"kind": "approval_queue", "id": dossier["mailbox"]},
        "payload": {
            "claimedSender": "unknown",
            "questionCode": "VERIFY_REQUEST",
            "referenceId": dossier["dossierId"],
        },
        "evidence": [line_ids[0]],
    }

@asynccontextmanager
async def lifespan(app):
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_id TEXT PRIMARY KEY,
            request_digest TEXT NOT NULL,
            input_digest TEXT NOT NULL,
            verifier_json TEXT NOT NULL,
            propose_response TEXT NOT NULL,
            commit_digest TEXT,
            commit_response TEXT
        );
        CREATE TABLE IF NOT EXISTS decisions (
            fingerprint TEXT PRIMARY KEY,
            proposal_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS receipts (
            evaluation_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY(evaluation_id, call_id)
        );
        """)
        await db.commit()
    yield


app = FastAPI(lifespan=lifespan)


async def connect():
    db = await aiosqlite.connect(DB_PATH, timeout=15)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA busy_timeout=15000")
    return db


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/mailroom/actions")
async def endpoint(request: Request):
    body_bytes = await request.body()
    if len(body_bytes) > MAX_BODY:
        raise HTTPException(413, "Request too large")

    try:
        body = json.loads(body_bytes)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Malformed JSON")

    if not isinstance(body, dict):
        raise HTTPException(422, "Expected JSON object")

    if body.get("profile") != PROFILE:
        raise HTTPException(422, "Invalid profile")

    if body.get("operation") == "propose":
        return await propose(body)
    if body.get("operation") == "commit":
        return await commit(body)
    raise HTTPException(400, "Invalid operation")


async def propose(body):
    required = {
        "profile", "operation", "evaluationId", "receiptVerifier",
        "corpus", "allowedActions", "dossiers",
    }
    if set(body) != required:
        raise HTTPException(422, "Invalid propose fields")

    if not isinstance(body["evaluationId"], str) or not body["evaluationId"]:
        raise HTTPException(422, "Invalid evaluation ID")

    verifier = body["receiptVerifier"]
    if (
        not isinstance(verifier, dict)
        or set(verifier) != {"algorithm", "publicKeyJwk"}
        or verifier["algorithm"] != "Ed25519"
    ):
        raise HTTPException(422, "Invalid receipt verifier")

    jwk = verifier["publicKeyJwk"]
    if (
        not isinstance(jwk, dict)
        or set(jwk) != {"kty", "crv", "x"}
        or jwk.get("kty") != "OKP"
        or jwk.get("crv") != "Ed25519"
        or not isinstance(jwk.get("x"), str)
    ):
        raise HTTPException(422, "Invalid public key JWK")

    import base64 as _b64
    try:
        _raw = _b64.urlsafe_b64decode(jwk["x"] + "=" * (-len(jwk["x"]) % 4))
    except Exception:
        raise HTTPException(422, "Invalid public key JWK")
    if len(_raw) != 32:
        raise HTTPException(422, "Invalid public key JWK")

    dossiers = body["dossiers"]
    if not isinstance(dossiers, list):
        raise HTTPException(422, "Invalid dossiers")

    ids = []
    for dossier in dossiers:
        if not isinstance(dossier, dict):
            raise HTTPException(422, "Invalid dossier")
        if set(dossier) != {
            "dossierId", "partition", "receivedAt", "mailbox",
            "objective", "sources",
        }:
            raise HTTPException(422, "Invalid dossier fields")

        ids.append(dossier["dossierId"])
        if not isinstance(dossier["sources"], list):
            raise HTTPException(422, "Invalid sources")

        for source in dossier["sources"]:
            if not isinstance(source, dict) or not isinstance(
                source.get("lines"), list
            ):
                raise HTTPException(422, "Invalid source")
            if set(source) != {
                "sourceId", "kind", "provenance", "title", "lines"
            }:
                raise HTTPException(422, "Invalid source fields")
            for line in source["lines"]:
                if (
                    not isinstance(line, dict)
                    or set(line) != {"lineId", "text"}
                    or not isinstance(line["lineId"], str)
                    or not isinstance(line["text"], str)
                ):
                    raise HTTPException(422, "Invalid evidence line")

    if len(ids) != len(set(ids)):
        raise HTTPException(422, "Duplicate dossier IDs")

    request_digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
    in_digest = input_digest(dossiers)
    eval_id = body["evaluationId"]

    db = await connect()
    try:
        await db.execute("BEGIN IMMEDIATE")
        old = await (
            await db.execute(
                "SELECT * FROM evaluations WHERE evaluation_id=?",
                (eval_id,),
            )
        ).fetchone()

        if old:
            if old["request_digest"] != request_digest:
                raise HTTPException(409, "Evaluation content conflict")
            response = json.loads(old["propose_response"])
            await db.commit()
            return JSONResponse(content=response)

        proposals = []
        for dossier in dossiers:
            fp = dossier_fingerprint(dossier)
            cached = await (
                await db.execute(
                    "SELECT proposal_json FROM decisions WHERE fingerprint=?",
                    (fp,),
                )
            ).fetchone()

            if cached:
                proposal = json.loads(cached["proposal_json"])
            
            else:
                try:
                    proposal = make_decision(dossier)
                    validate_proposal(proposal, dossier)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"DECISION ERROR for {dossier['dossierId']}: "
                          f"{type(exc).__name__}: {exc}")
                    proposal = _fallback_proposal(dossier)
                await db.execute(
                    "INSERT OR IGNORE INTO decisions VALUES (?, ?)",
                    (fp, canonical_bytes(proposal).decode()),
                )
                stored = await (
                    await db.execute(
                        "SELECT proposal_json FROM decisions WHERE fingerprint=?",
                        (fp,),
                    )
                ).fetchone()
                proposal = json.loads(stored["proposal_json"])

            # Call IDs are dossier-specific and stable across evaluations.
            proposal["callId"] = stable_call_id(dossier)
            validate_proposal(proposal, dossier)
            proposals.append(proposal)

        if len({p["callId"] for p in proposals}) != len(proposals):
            raise HTTPException(422, "Duplicate call IDs")

        response = {
            "profile": PROFILE,
            "evaluationId": eval_id,
            "status": "awaiting_receipts",
            "inputDigest": in_digest,
            "proposals": proposals,
        }
        encoded = canonical_bytes(response).decode()
        if len(encoded.encode()) > 512 * 1024:
            raise HTTPException(422, "Response too large")

        await db.execute(
            """INSERT INTO evaluations
            (evaluation_id, request_digest, input_digest, verifier_json,
             propose_response)
            VALUES (?, ?, ?, ?, ?)""",
            (
                eval_id, request_digest, in_digest,
                canonical_bytes(verifier).decode(), encoded,
            ),
        )
        await db.commit()
        return JSONResponse(content=response)
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        import traceback
        traceback.print_exc()          # prints full stack to Render logs
        print(f"PROPOSE ERROR: {type(e).__name__}: {exc}")
        raise HTTPException(503, "Proposal generation failed")
    finally:
        await db.close()


async def commit(body):
    required = {
        "profile", "operation", "evaluationId", "inputDigest", "receipts"
    }
    if set(body) != required:
        raise HTTPException(422, "Invalid commit fields")
    if body["operation"] != "commit":
        raise HTTPException(400, "Invalid operation")
    if not isinstance(body["receipts"], list):
        raise HTTPException(422, "Invalid receipts")

    request_digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
    db = await connect()

    try:
        await db.execute("BEGIN IMMEDIATE")
        row = await (
            await db.execute(
                "SELECT * FROM evaluations WHERE evaluation_id=?",
                (body["evaluationId"],),
            )
        ).fetchone()

        if row is None:
            raise HTTPException(422, "Unknown evaluation")

        if row["commit_digest"] is not None:
            if row["commit_digest"] != request_digest:
                raise HTTPException(409, "Commit content conflict")
            saved = json.loads(row["commit_response"])
            await db.commit()
            return JSONResponse(content=saved)

        if body["inputDigest"] != row["input_digest"]:
            raise HTTPException(422, "Input digest mismatch")

        saved_propose = json.loads(row["propose_response"])
        proposals = saved_propose["proposals"]
        receipts = body["receipts"]

        if len(receipts) != len(proposals):
            raise HTTPException(422, "Receipt count mismatch")

        expected = {p["callId"]: p for p in proposals}
        if len(expected) != len(proposals):
            raise HTTPException(422, "Duplicate stored call ID")

        seen = set()
        verified = []
        verifier = json.loads(row["verifier_json"])["publicKeyJwk"]

        for receipt in receipts:
            keys = {
                "dossierId", "callId", "action", "accepted",
                "proposalDigest", "receiptId", "receiptSignature",
            }
            if not isinstance(receipt, dict) or set(receipt) != keys:
                raise HTTPException(422, "Invalid receipt schema")

            call_id = receipt["callId"]
            if not isinstance(call_id, str) or call_id in seen:
                raise HTTPException(422, "Duplicate receipt call ID")
            seen.add(call_id)

            proposal = expected.get(call_id)
            if proposal is None:
                raise HTTPException(422, "Unknown receipt call ID")

            if (
                receipt["dossierId"] != proposal["dossierId"]
                or receipt["action"] != proposal["action"]
            ):
                raise HTTPException(422, "Receipt scope mismatch")

            if type(receipt["accepted"]) is not bool:
                raise HTTPException(422, "Invalid accepted value")

            for field in ("proposalDigest", "receiptId", "receiptSignature"):
                if not isinstance(receipt[field], str) or not receipt[field]:
                    raise HTTPException(422, "Missing receipt field")

            if receipt["proposalDigest"] != proposal_digest(proposal):
                raise HTTPException(422, "Proposal digest mismatch")

            if not verify_receipt_signature(
                verifier, body["evaluationId"], row["input_digest"], receipt
            ):
                raise HTTPException(422, "Invalid receipt signature")

            verified.append((proposal, receipt))

        if seen != set(expected):
            raise HTTPException(422, "Missing receipt")

        outcomes = []
        for proposal, receipt in verified:
            await db.execute(
                "INSERT INTO receipts VALUES (?, ?, ?)",
                (
                    body["evaluationId"], proposal["callId"],
                    canonical_bytes(receipt).decode(),
                ),
            )
            outcomes.append({
                "dossierId": proposal["dossierId"],
                "callId": proposal["callId"],
                "action": proposal["action"],
                "proposalDigest": receipt["proposalDigest"],
                "receiptId": receipt["receiptId"],
                "status": "executed" if receipt["accepted"] else "rejected",
            })

        response = {
            "profile": PROFILE,
            "evaluationId": body["evaluationId"],
            "status": "completed",
            "inputDigest": row["input_digest"],
            "outcomes": outcomes,
        }
        await db.execute(
            """UPDATE evaluations
            SET commit_digest=?, commit_response=? WHERE evaluation_id=?""",
            (
                request_digest, canonical_bytes(response).decode(),
                body["evaluationId"],
            ),
        )
        await db.commit()
        return JSONResponse(content=response)
    except HTTPException:
        await db.rollback()
        raise
    except Exception as exc:
        await db.rollback()
        import traceback
        traceback.print_exc()
        print(f"COMMIT ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(503, "Commit failed")
    finally:
        await db.close()
