import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

DB_PATH = Path("data/agent.db")
DB_PATH.parent.mkdir(exist_ok=True)

MODEL_BASE_URL = os.environ["MODEL_BASE_URL"].rstrip("/")
MODEL_API_KEY = os.environ["MODEL_API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
MODEL_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "9"))

app = FastAPI()
db_lock = threading.Lock()
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL,
  state_json TEXT NOT NULL
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS receipts (
  receipt_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  receipt_hash TEXT NOT NULL
)
""")
conn.commit()


def now_ns():
    return str(time.time_ns())


def hex_id(n):
    return uuid.uuid4().hex[:n]


def canonical_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def attr(key, value):
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def get_run(run_id):
    with db_lock:
        row = conn.execute(
            "SELECT request_hash, state_json FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
    if not row:
        return None, None
    return row[0], json.loads(row[1])


def save_run(run_id, request_hash, state):
    with db_lock:
        conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, request_hash, state_json) VALUES(?,?,?)",
            (run_id, request_hash, json.dumps(state)),
        )
        conn.commit()


def receipt_exists(receipt_id):
    with db_lock:
        row = conn.execute(
            "SELECT receipt_hash FROM receipts WHERE receipt_id=?", (receipt_id,)
        ).fetchone()
    return row[0] if row else None


def save_receipt(receipt_id, run_id, receipt_hash):
    with db_lock:
        conn.execute(
            "INSERT INTO receipts(receipt_id, run_id, receipt_hash) VALUES(?,?,?)",
            (receipt_id, run_id, receipt_hash),
        )
        conn.commit()


def parse_traceparent(value):
    if not value:
        return None, None, None
    m = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})", value)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


async def call_model(incident, tool_catalog, policy):
    catalog = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "inputSchema": t.get("inputSchema", {}),
        }
        for t in tool_catalog
    ]

    prompt = f"""Choose the single most likely root cause.

Return ONLY compact JSON:
{{
  "rootCause": "<one exact value from allowedRootCauses>",
  "evidence": ["<2-4 evidence IDs>"],
  "diagnostics": [
    {{"toolName":"<tool name>","arguments":{{}}}}
  ]
}}

Rules:
- rootCause must be exactly one allowedRootCauses value.
- evidence must contain 2-4 IDs that appear in the transcript.
- Use at most {policy.get('maximumDiagnostics', 3)} diagnostics.
- Prefer independent diagnostic calls that can run in parallel.
- Use exact incident-specific arguments.
- Do not choose an effect tool here.

allowedRootCauses:
{json.dumps(incident['allowedRootCauses'])}

toolCatalog:
{json.dumps(catalog)}

incident:
{json.dumps(incident)}
"""

    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 900,
    }

    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {MODEL_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(content)

    allowed = set(incident["allowedRootCauses"])
    if parsed.get("rootCause") not in allowed:
        parsed["rootCause"] = incident["allowedRootCauses"][0]

    evidence = [e for e in parsed.get("evidence", []) if isinstance(e, str)][:4]
    parsed["evidence"] = evidence

    diagnostics = []
    max_diag = int(policy.get("maximumDiagnostics", 3))
    for d in parsed.get("diagnostics", [])[:max_diag]:
        if not isinstance(d, dict):
            continue
        name = d.get("toolName")
        args = d.get("arguments", {})
        if isinstance(name, str) and isinstance(args, dict):
            diagnostics.append({"toolName": name, "arguments": args})

    return parsed, content


def make_dispatch(action, phase, evidence, attempt, trace_id, client_span_id):
    return {
        "actionId": action["actionId"],
        "callId": action["callId"],
        "phase": phase,
        "toolName": action["toolName"],
        "arguments": action["arguments"],
        "evidence": evidence,
        "attempt": attempt,
        "traceparent": f"00-{trace_id}-{client_span_id}-01",
    }


def choose_effect(tool_catalog, policy, root_cause):
    effect_tools = policy.get("effectTools", [])
    for tool in effect_tools:
        return tool
    return None


def build_final(state):
    run_id = state["runId"]
    marker = state["publicMarker"]
    trace_id = state["traceId"]
    server_span = state["serverSpanId"]
    agent_span = state["agentSpanId"]

    spans = []

    def span(name, kind, parent, start, end, attrs, links=None, status=None):
        s = {
            "traceId": trace_id,
            "spanId": hex_id(16),
            "name": name,
            "kind": kind,
            "startTimeUnixNano": start,
            "endTimeUnixNano": end,
            "attributes": [attr("ga5.run.id", run_id), attr("ga5.public.marker", marker)] + attrs,
        }
        if parent:
            s["parentSpanId"] = parent
        if links:
            s["links"] = links
        if status:
            s["status"] = status
        spans.append(s)
        return s["spanId"]

    start = state["startedNs"]
    end = now_ns()

    span(
        "POST /v2/incidents",
        2,
        None,
        start,
        end,
        [attr("http.request.method", "POST")],
    )

    span(
        "invoke_agent incident-response",
        1,
        server_span,
        start,
        end,
        [],
    )

    span(
        "chat incident-plan",
        3,
        agent_span,
        state["modelStartedNs"],
        state["modelEndedNs"],
        [
            attr("gen_ai.operation.name", "chat"),
            attr("gen_ai.request.model", MODEL_NAME),
        ],
    )

    diagnostic_spans = []

    for action in state["actions"].values():
        if action["kind"] != "tool":
            continue

        tool_span_id = hex_id(16)
        tool_start = action["createdNs"]
        tool_end = action.get("completedNs", end)

        span(
            f"execute_tool {action['toolName']}",
            1,
            agent_span,
            tool_start,
            tool_end,
            [
                attr("ga5.action.id", action["actionId"]),
                attr("gen_ai.tool.name", action["toolName"]),
                attr("gen_ai.tool.call.id", action["callId"]),
                attr("gen_ai.operation.name", "execute_tool"),
            ],
        )

        for attempt in action["attempts"]:
            client_span_id = attempt["clientSpanId"]
            status = attempt.get("status", 0)
            error_type = attempt.get("errorType")

            attrs = [
                attr("ga5.action.id", action["actionId"]),
                attr("ga5.attempt", attempt["attempt"]),
                attr("ga5.receipt.id", attempt["receiptId"]),
                attr("ga5.receipt.nonce", attempt["nonce"]),
                attr("http.request.method", "POST"),
                attr("http.request.resend_count", attempt["attempt"] - 1),
                attr("http.response.status_code", status),
            ]

            span_status = None
            if error_type == "timeout" or status == 503:
                span_status = {"code": 2}
                attrs.append(attr("error.type", error_type or "503"))

            span(
                f"POST tool/{action['toolName']}",
                3,
                tool_span_id,
                attempt["startedNs"],
                attempt.get("endedNs", end),
                attrs,
                status=span_status,
            )

            if action["phase"] == "diagnostic":
                diagnostic_spans.append(tool_span_id)

    if len(diagnostic_spans) > 1:
        span(
            "incident.join",
            1,
            agent_span,
            state.get("joinStartedNs", end),
            end,
            [],
            links=[
                {"traceId": trace_id, "spanId": sid}
                for sid in diagnostic_spans
            ],
        )

    if state.get("approval"):
        approval = state["approval"]
        span(
            "approval_gate",
            1,
            agent_span,
            approval["createdNs"],
            approval.get("decidedNs", end),
            [
                attr("ga5.approval.id", approval["approvalId"]),
                attr("ga5.approval.receipt.nonce", approval.get("nonce", "")),
            ],
        )

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        attr("service.name", "incident-response"),
                        attr("ga5.run.id", run_id),
                        attr("ga5.public.marker", marker),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "ga5-incident-agent", "version": "1"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def public_state(state):
    if state["status"] in ("completed", "failed"):
        return {
            "runId": state["runId"],
            "status": state["status"],
            "diagnosis": state["diagnosis"],
            "chosenEffect": state.get("chosenEffect"),
            "suppressed": state.get("suppressed", []),
            "actionLog": state["actionLog"],
            "receiptLog": state["receiptLog"],
            "otlp": build_final(state),
        }

    pending = []
    for action in state["actions"].values():
        if action["awaitingRetry"]:
            pending.append(
                make_dispatch(
                    action,
                    action["phase"],
                    action["evidence"],
                    action["nextAttempt"],
                    state["traceId"],
                    action["nextClientSpanId"],
                )
            )

    if state["pendingEffect"]:
        effect = state["pendingEffect"]
        pending.append(
            make_dispatch(
                effect,
                "effect",
                effect["evidence"],
                1,
                state["traceId"],
                effect["clientSpanId"],
            )
        )

    approvals = []
    if state.get("approvalRequest"):
        approvals.append(state["approvalRequest"])

    return {
        "runId": state["runId"],
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": pending,
        "approvals": approvals,
    }


@app.post("/v2/incidents")
async def create_incident(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    if body.get("profile") != "ga5-incident-agent/v2":
        return JSONResponse({"error": "unsupported profile"}, status_code=400)

    run_id = body.get("runId")
    if not isinstance(run_id, str) or not run_id:
        return JSONResponse({"error": "invalid runId"}, status_code=400)

    request_hash = canonical_hash(body)
    existing_hash, state = get_run(run_id)

    if state:
        if existing_hash != request_hash:
            return JSONResponse({"error": "runId content changed"}, status_code=409)
        return JSONResponse(public_state(state))

    incident = body["incident"]
    tool_catalog = body["toolCatalog"]
    policy = body["policy"]

    incoming_tp = request.headers.get("traceparent")
    trace_id, _parent, flags = parse_traceparent(incoming_tp)
    if not trace_id:
        trace_id = hex_id(32)
        flags = "01"

    server_span_id = hex_id(16)
    agent_span_id = hex_id(16)
    model_started = time.time_ns()

    try:
        plan, _raw = await call_model(incident, tool_catalog, policy)
    except Exception:
        allowed = incident.get("allowedRootCauses", ["unknown"])
        plan = {
            "rootCause": allowed[0],
            "evidence": [],
            "diagnostics": [],
        }

    model_ended = time.time_ns()

    evidence = plan.get("evidence", [])
    if len(evidence) < 2:
        ids = re.findall(r"\[(ev_[A-Za-z0-9_-]+)\]", incident.get("transcript", ""))
        evidence = (evidence + ids)[:4]
    evidence = list(dict.fromkeys(evidence))[:4]

    state = {
        "runId": run_id,
        "publicMarker": body.get("publicMarker", ""),
        "incident": incident,
        "toolCatalog": tool_catalog,
        "policy": policy,
        "traceId": trace_id,
        "traceFlags": flags or "01",
        "serverSpanId": server_span_id,
        "agentSpanId": agent_span_id,
        "startedNs": str(model_started),
        "modelStartedNs": str(model_started),
        "modelEndedNs": str(model_ended),
        "diagnosis": {
            "rootCause": plan["rootCause"],
            "evidence": evidence,
        },
        "actions": {},
        "receiptLog": [],
        "actionLog": [],
        "status": "waiting",
        "suppressed": [],
        "pendingEffect": None,
        "approval": None,
        "approvalRequest": None,
    }

    diagnostic_tools = {
        t["name"]: t
        for t in tool_catalog
        if t["name"] not in policy.get("effectTools", [])
    }

    for i, diagnostic in enumerate(plan.get("diagnostics", [])):
        name = diagnostic["toolName"]
        if name not in diagnostic_tools:
            continue

        action_id = f"diag-{run_id[:8]}-{i}"
        call_id = action_id
        client_span_id = hex_id(16)

        action = {
            "kind": "tool",
            "phase": "diagnostic",
            "actionId": action_id,
            "callId": call_id,
            "toolName": name,
            "arguments": diagnostic["arguments"],
            "evidence": evidence[:1],
            "attempts": [],
            "awaitingRetry": False,
            "nextAttempt": 1,
            "nextClientSpanId": client_span_id,
            "createdNs": str(time.time_ns()),
            "completedNs": None,
            "dependentEffect": None,
        }
        state["actions"][action_id] = action

    if not state["actions"]:
        state["status"] = "failed"

    save_run(run_id, request_hash, state)
    return JSONResponse(public_state(state))


@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipt(run_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    receipt_id = body.get("receiptId")
    if not isinstance(receipt_id, str) or not receipt_id:
        return JSONResponse({"error": "invalid receiptId"}, status_code=400)

    receipt_hash = canonical_hash(body)
    existing = receipt_exists(receipt_id)

    _request_hash, state = get_run(run_id)
    if not state:
        return JSONResponse({"error": "unknown run"}, status_code=404)

    if existing:
        if existing != receipt_hash:
            return JSONResponse({"error": "receipt content changed"}, status_code=409)
        return JSONResponse(public_state(state))

    save_receipt(receipt_id, run_id, receipt_hash)

    if "outcomes" in body:
        for outcome in body["outcomes"]:
            apply_outcome(state, receipt_id, outcome)
    elif "approvals" in body:
        for approval in body["approvals"]:
            apply_approval(state, receipt_id, approval)
    else:
        return JSONResponse({"error": "invalid receipt"}, status_code=400)

    maybe_finish(state)
    save_run(run_id, state.get("requestHash", ""), state)
    return JSONResponse(public_state(state))


def apply_outcome(state, receipt_id, outcome):
    action = state["actions"].get(outcome.get("actionId"))
    if not action:
        return

    attempt_no = int(outcome.get("attempt", 1))
    status = outcome.get("status", 0)
    error_type = outcome.get("errorType")
    nonce = outcome.get("nonce", "")

    attempt = {
        "attempt": attempt_no,
        "receiptId": receipt_id,
        "nonce": nonce,
        "status": status,
        "errorType": error_type,
        "startedNs": action["createdNs"],
        "endedNs": str(time.time_ns()),
        "clientSpanId": action["nextClientSpanId"],
    }
    action["attempts"].append(attempt)
    state["receiptLog"].append(
        {
            "receiptId": receipt_id,
            "actionId": action["actionId"],
            "callId": action["callId"],
            "attempt": attempt_no,
            "status": status,
            "resultClass": outcome.get("resultClass", ""),
            "nonce": nonce,
        }
    )

    if error_type == "timeout":
        action["awaitingRetry"] = False
        action["completedNs"] = str(time.time_ns())
        if action["dependentEffect"]:
            state["suppressed"].append(action["dependentEffect"])
        return

    if status == 503 and attempt_no == 1:
        action["awaitingRetry"] = True
        action["nextAttempt"] = 2
        action["nextClientSpanId"] = hex_id(16)
        return

    action["awaitingRetry"] = False
    action["completedNs"] = str(time.time_ns())

    if action["phase"] == "diagnostic":
        state.setdefault("joinStartedNs", str(time.time_ns()))
        start_effect_if_ready(state)


def start_effect_if_ready(state):
    diagnostics = [
        a for a in state["actions"].values() if a["phase"] == "diagnostic"
    ]

    if not diagnostics:
        return

    if any(a["awaitingRetry"] for a in diagnostics):
        return

    if any(not a["attempts"] for a in diagnostics):
        return

    if state.get("pendingEffect") or state.get("approval"):
        return

    if any(
        a["attempts"][-1].get("errorType") == "timeout"
        for a in diagnostics
    ):
        return

    effect_name = choose_effect(
        state["toolCatalog"], state["policy"], state["diagnosis"]["rootCause"]
    )
    if not effect_name:
        state["status"] = "completed"
        return

    effect_id = f"effect-{state['runId'][:8]}"
    client_span_id = hex_id(16)

    effect = {
        "kind": "tool",
        "phase": "effect",
        "actionId": effect_id,
        "callId": effect_id,
        "toolName": effect_name,
        "arguments": {},
        "evidence": state["diagnosis"]["evidence"][:1],
        "attempts": [],
        "awaitingRetry": False,
        "nextAttempt": 1,
        "nextClientSpanId": client_span_id,
        "createdNs": str(time.time_ns()),
        "completedNs": None,
        "clientSpanId": client_span_id,
    }

    state["chosenEffect"] = effect_name

    if effect_name in state["policy"].get("approvalRequiredFor", []):
        approval_id = f"approval-{state['runId'][:8]}"
        digest = hashlib.sha256(
            json.dumps(effect["arguments"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        state["approvalRequest"] = {
            "approvalId": approval_id,
            "actionId": effect_id,
            "toolName": effect_name,
            "argumentsDigest": digest,
        }
        state["approval"] = {
            "approvalId": approval_id,
            "actionId": effect_id,
            "toolName": effect_name,
            "arguments": effect["arguments"],
            "createdNs": str(time.time_ns()),
            "decidedNs": None,
            "nonce": None,
        }
        state["pendingEffect"] = None
    else:
        state["pendingEffect"] = effect


def apply_approval(state, receipt_id, approval):
    pending = state.get("approvalRequest")
    if not pending:
        return

    if approval.get("approvalId") != pending["approvalId"]:
        return

    if approval.get("decision") != "approved":
        state["status"] = "failed"
        return

    nonce = approval.get("nonce", "")
    state["approval"]["decidedNs"] = str(time.time_ns())
    state["approval"]["nonce"] = nonce

    state["receiptLog"].append(
        {
            "receiptId": receipt_id,
            "approvalId": pending["approvalId"],
            "decision": "approved",
            "nonce": nonce,
        }
    )

    effect = {
        "kind": "tool",
        "phase": "effect",
        "actionId": pending["actionId"],
        "callId": pending["actionId"],
        "toolName": pending["toolName"],
        "arguments": state["approval"]["arguments"],
        "evidence": state["diagnosis"]["evidence"][:1],
        "attempts": [],
        "awaitingRetry": False,
        "nextAttempt": 1,
        "nextClientSpanId": hex_id(16),
        "createdNs": str(time.time_ns()),
        "completedNs": None,
        "clientSpanId": hex_id(16),
        "approvalId": pending["approvalId"],
        "approvalNonce": nonce,
    }

    state["pendingEffect"] = effect
    state["approvalRequest"] = None


def maybe_finish(state):
    effect = state.get("pendingEffect")
    if effect and effect["attempts"]:
        last = effect["attempts"][-1]
        if last.get("errorType") == "timeout":
            state["suppressed"].append(effect["actionId"])
            state["pendingEffect"] = None
            state["status"] = "failed"
            return

        if last.get("status") == 503 and last["attempt"] == 1:
            effect["awaitingRetry"] = True
            effect["nextAttempt"] = 2
            effect["nextClientSpanId"] = hex_id(16)
            return

        state["pendingEffect"] = None
        state["status"] = "completed"
        return

    if state.get("approvalRequest"):
        return

    diagnostics = [
        a for a in state["actions"].values() if a["phase"] == "diagnostic"
    ]

    if diagnostics and all(
        a["attempts"]
        and not a["awaitingRetry"]
        for a in diagnostics
    ):
        start_effect_if_ready(state)


@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    _request_hash, state = get_run(run_id)
    if not state:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    return JSONResponse(public_state(state))
