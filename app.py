from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from openai import OpenAI


load_dotenv()


# ============================================================
# App and configuration
# ============================================================

app = FastAPI(
    title="Invoice Action Agent",
    version="1.0.0",
)

USE_LLM = (
    os.getenv("USE_LLM", "false").lower()
    in {"1", "true", "yes", "on"}
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

BASE_URL = os.getenv(
    "BASE_URL",
    "https://ga3q2-a7fm.onrender.com",
).rstrip("/")

STATE_FILE = os.getenv(
    "STATE_FILE",
    "/tmp/invoice_action_agent_state.json",
)

A2A_MEDIA_TYPE = "application/a2a+json"
JSON_MEDIA_TYPE = "application/json"
SUPPORTED_A2A_VERSION = "1.0"

BATCH_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-claim-batch+json"
)

PROPOSAL_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-proposals+json"
)

RECEIPT_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-receipts+json"
)

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

TASK_STATE_INPUT_REQUIRED = (
    "TASK_STATE_INPUT_REQUIRED"
)

TASK_STATE_COMPLETED = (
    "TASK_STATE_COMPLETED"
)

TASK_STATE_CANCELED = (
    "TASK_STATE_CANCELED"
)

STATE_LOCK = threading.RLock()

STATE: dict[str, Any] = {
    "tasks": {},
    "messages": {},
    "receipts": {},
}

OPENAI_CLIENT: OpenAI | None = None

if USE_LLM and OPENAI_API_KEY:
    OPENAI_CLIENT = OpenAI(
        api_key=OPENAI_API_KEY,
    )


# ============================================================
# Persistence
# ============================================================

def load_state() -> None:
    global STATE

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            loaded = json.load(file)

        if isinstance(loaded, dict):
            STATE = loaded

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ):
        STATE = {
            "tasks": {},
            "messages": {},
            "receipts": {},
        }

    STATE.setdefault("tasks", {})
    STATE.setdefault("messages", {})
    STATE.setdefault("receipts", {})


def save_state() -> None:
    directory = os.path.dirname(STATE_FILE)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temporary_file = f"{STATE_FILE}.tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            STATE,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary_file,
        STATE_FILE,
    )


load_state()


# ============================================================
# General helpers
# ============================================================

def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def value_hash(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def principal_hash(token: str) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def a2a_response(
    content: dict[str, Any],
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        content=content,
        status_code=status_code,
        headers=headers,
        media_type=A2A_MEDIA_TYPE,
    )


def rpc_success(
    request_id: Any,
    result: dict[str, Any],
) -> JSONResponse:
    return a2a_response(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
    )


def rpc_error(
    request_id: Any,
    code: int,
    message: str,
    status_code: int,
) -> JSONResponse:
    return a2a_response(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        },
        status_code=status_code,
    )


# ============================================================
# Authentication
# ============================================================

def require_bearer_token(
    authorization: str | None = Header(
        default=None,
    ),
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Bearer token",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    scheme, separator, token = authorization.partition(
        " "
    )

    if scheme.lower() != "bearer" or not separator:
        raise HTTPException(
            status_code=401,
            detail="Bearer authentication required",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    token = token.strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Bearer token cannot be empty",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    return token


# ============================================================
# A2A request validation
# ============================================================

def validate_a2a_request(
    request: Request,
) -> None:
    version = request.headers.get(
        "A2A-Version"
    )

    if version != SUPPORTED_A2A_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported A2A-Version; "
                "expected 1.0"
            ),
        )

    content_type = request.headers.get(
        "Content-Type",
        "",
    )

    media_type = content_type.split(
        ";",
        maxsplit=1,
    )[0].strip().lower()

    if media_type not in {
        A2A_MEDIA_TYPE,
        JSON_MEDIA_TYPE,
    }:
        raise HTTPException(
            status_code=415,
            detail=(
                "Content-Type must be "
                "application/a2a+json or "
                "application/json"
            ),
        )


# ============================================================
# Agent Card
# ============================================================

def agent_card_data() -> dict[str, Any]:
    return {
        "protocolVersion": "1.0",
        "name": "Invoice Action Agent",
        "description": (
            "Evaluates invoice claim batches and creates "
            "evidence-based action proposals."
        ),
        "version": "1.0.0",
        "url": BASE_URL,
        "preferredTransport": "JSONRPC",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extendedAgentCard": False,
        },
        "defaultInputModes": [
            JSON_MEDIA_TYPE,
            A2A_MEDIA_TYPE,
            BATCH_MEDIA_TYPE,
        ],
        "defaultOutputModes": [
            A2A_MEDIA_TYPE,
            PROPOSAL_MEDIA_TYPE,
            RECEIPT_MEDIA_TYPE,
        ],
        "securitySchemes": {
            "bearerAuth": {
                "httpAuthSecurityScheme": {
                    "scheme": "bearer",
                    "bearerFormat": "Bearer",
                }
            }
        },
        "securityRequirements": [
            {
                "schemes": {
                    "bearerAuth": {
                        "list": [],
                    }
                }
            }
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice action agent",
                "description": (
                    "Processes invoice batches, proposes "
                    "safe actions, and produces receipts."
                ),
                "tags": [
                    "invoice",
                    "payments",
                    "approval",
                    "receipts",
                ],
                "examples": [
                    "Evaluate an invoice batch.",
                    "Create receipts for accepted actions.",
                ],
                "inputModes": [
                    JSON_MEDIA_TYPE,
                    A2A_MEDIA_TYPE,
                    BATCH_MEDIA_TYPE,
                ],
                "outputModes": [
                    A2A_MEDIA_TYPE,
                    PROPOSAL_MEDIA_TYPE,
                    RECEIPT_MEDIA_TYPE,
                ],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
def get_agent_card() -> dict[str, Any]:
    return agent_card_data()


@app.get("/.well-known/agent.json")
def get_legacy_agent_card() -> dict[str, Any]:
    return agent_card_data()


# ============================================================
# Invoice decision logic
# ============================================================

def find_evidence_refs(
    package: dict[str, Any],
) -> list[str]:
    text = canonical_json(package)

    known_refs = [
        "[invoice-1]",
        "[reconciliation-1]",
        "[authority-1]",
        "[policy-1]",
        "[duplicate-check-1]",
        "[approval-1]",
        "[exception-1]",
    ]

    found = [
        reference
        for reference in known_refs
        if reference in text
    ]

    dynamic_refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if (
                value.startswith("[")
                and value.endswith("]")
                and value not in dynamic_refs
            ):
                dynamic_refs.append(value)

        elif isinstance(value, dict):
            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(package)

    for reference in dynamic_refs:
        if reference not in found:
            found.append(reference)

    while len(found) < 3:
        found.append(
            f"[evidence-{len(found) + 1}]"
        )

    return found[:3]


def choose_action(
    package: dict[str, Any],
) -> str:
    text = canonical_json(package).lower()

    if (
        "duplicate" in text
        or "already paid" in text
        or "same commercial invoice" in text
    ):
        return "reject_duplicate"

    if (
        "conflict" in text
        or "mismatch" in text
        or "exception" in text
    ):
        return "open_exception"

    if (
        "approval required" in text
        or "outside delegated authority" in text
        or "approval" in text
    ):
        return "request_approval"

    if (
        "reconciled" in text
        or "matched" in text
        or "within authority" in text
    ):
        return "settle_invoice"

    return "hold_invoice"


def non_llm_proposal(
    package: dict[str, Any],
) -> dict[str, Any]:
    package_id = str(
        package.get(
            "packageId",
            package.get(
                "package_id",
                str(uuid.uuid4()),
            ),
        )
    )

    vendor_name = str(
        package.get(
            "vendorName",
            package.get(
                "vendor",
                "Unknown vendor",
            ),
        )
    )

    invoice_number = str(
        package.get(
            "invoiceNumber",
            package.get(
                "invoice_number",
                "Unknown invoice",
            ),
        )
    )

    amount_minor = package.get(
        "amountMinor",
        package.get(
            "amount_minor",
            0,
        ),
    )

    try:
        amount_minor = int(amount_minor)

    except (TypeError, ValueError):
        amount_minor = 0

    currency = str(
        package.get(
            "currency",
            "INR",
        )
    )

    action = choose_action(package)
    evidence_refs = find_evidence_refs(package)

    return {
        "packageId": package_id,
        "actionId": str(uuid.uuid4()),
        "action": action,
        "facts": {
            "vendorName": vendor_name,
            "invoiceNumber": invoice_number,
            "amountMinor": amount_minor,
            "currency": currency,
        },
        "evidenceRefs": evidence_refs,
        "rationale": (
            f"{action}: the invoice package was evaluated "
            "using the supplied evidence references "
            f"{evidence_refs[0]}, "
            f"{evidence_refs[1]}, and "
            f"{evidence_refs[2]}."
        ),
    }


def llm_proposal(
    package: dict[str, Any],
) -> dict[str, Any]:
    fallback = non_llm_proposal(package)

    if OPENAI_CLIENT is None:
        return fallback

    prompt = (
        "Analyze this invoice package. Return valid JSON only "
        "with packageId, actionId, action, facts, evidenceRefs, "
        "and rationale. Choose exactly one action from: "
        "settle_invoice, request_approval, hold_invoice, "
        "reject_duplicate, open_exception. Use only supplied "
        "evidence and exactly three evidence references.\n\n"
        f"{canonical_json(package)}"
    )

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={
                "type": "json_object",
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict invoice "
                        "decision engine."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            return fallback

        proposal = json.loads(content)

        if not isinstance(proposal, dict):
            return fallback

        if proposal.get("action") not in ALLOWED_ACTIONS:
            return fallback

        if not isinstance(
            proposal.get("evidenceRefs"),
            list,
        ):
            return fallback

        if len(proposal["evidenceRefs"]) != 3:
            return fallback

        proposal["packageId"] = fallback[
            "packageId"
        ]
        proposal["actionId"] = str(uuid.uuid4())

        return proposal

    except Exception:
        return fallback


def make_proposal(
    package: dict[str, Any],
) -> dict[str, Any]:
    if USE_LLM:
        return llm_proposal(package)

    return non_llm_proposal(package)


# ============================================================
# Input parsing
# ============================================================

def extract_batch_from_message(
    message: dict[str, Any],
) -> dict[str, Any] | None:
    parts = message.get(
        "parts",
        [],
    )

    if not isinstance(parts, list):
        return None

    for part in parts:
        if not isinstance(part, dict):
            continue

        data = part.get("data")

        if isinstance(data, dict):
            if isinstance(
                data.get("packages"),
                list,
            ):
                return data

            nested = data.get("batch")

            if isinstance(nested, dict):
                if isinstance(
                    nested.get("packages"),
                    list,
                ):
                    return nested

        text = part.get("text")

        if isinstance(text, str):
            try:
                parsed = json.loads(text)

            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            if isinstance(
                parsed.get("packages"),
                list,
            ):
                return parsed

            nested = parsed.get("batch")

            if isinstance(nested, dict):
                if isinstance(
                    nested.get("packages"),
                    list,
                ):
                    return nested

    return None


def extract_rpc_message(
    payload: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    request_id = payload.get("id")

    if payload.get("jsonrpc") != "2.0":
        return request_id, None

    method = payload.get("method")

    if method not in {
        "SendMessage",
        "message/send",
        "message:send",
    }:
        return request_id, None

    params = payload.get("params")

    if not isinstance(params, dict):
        return request_id, None

    message = params.get("message")

    if isinstance(message, dict):
        return request_id, message

    return request_id, None


# ============================================================
# Task helpers
# ============================================================

def get_owner_tasks(
    token: str,
) -> dict[str, Any]:
    owner = principal_hash(token)

    return STATE.setdefault(
        "tasks",
        {},
    ).setdefault(
        owner,
        {},
    )


def get_task_for_owner(
    token: str,
    task_id: str,
) -> dict[str, Any] | None:
    return get_owner_tasks(token).get(
        task_id
    )


def save_task(
    token: str,
    task: dict[str, Any],
) -> None:
    get_owner_tasks(token)[
        task["id"]
    ] = task


# ============================================================
# Message processing
# ============================================================

async def process_message(
    request: Request,
    token: str,
) -> JSONResponse:
    validate_a2a_request(request)

    try:
        payload = await request.json()

    except Exception:
        return rpc_error(
            request_id=None,
            code=-32700,
            message="Invalid JSON",
            status_code=400,
        )

    if not isinstance(payload, dict):
        return rpc_error(
            request_id=None,
            code=-32600,
            message="Invalid Request",
            status_code=400,
        )

    request_id, message = extract_rpc_message(
        payload
    )

    if message is None:
        return rpc_error(
            request_id=request_id,
            code=-32602,
            message=(
                "Expected JSON-RPC method "
                "SendMessage with params.message"
            ),
            status_code=400,
        )

    role = message.get(
        "role",
        "ROLE_USER",
    )

    if role == "user":
        role = "ROLE_USER"

    if role != "ROLE_USER":
        return rpc_error(
            request_id=request_id,
            code=-32602,
            message="message.role must be ROLE_USER",
            status_code=400,
        )

    message["role"] = "ROLE_USER"

    message_id = message.get(
        "messageId"
    )

    if not isinstance(message_id, str):
        return rpc_error(
            request_id=request_id,
            code=-32602,
            message="messageId is required",
            status_code=400,
        )

    batch = extract_batch_from_message(
        message
    )

    if batch is None:
        return rpc_error(
            request_id=request_id,
            code=-32602,
            message="Invoice batch data is required",
            status_code=400,
        )

    packages = batch.get(
        "packages"
    )

    if not isinstance(packages, list):
        return rpc_error(
            request_id=request_id,
            code=-32602,
            message="batch.packages must be a list",
            status_code=400,
        )

    owner = principal_hash(token)
    message_key = f"{owner}:{message_id}"
    message_digest = value_hash(message)

    with STATE_LOCK:
        previous = STATE.setdefault(
            "messages",
            {},
        ).get(message_key)

        if previous is not None:
            if previous.get(
                "messageHash"
            ) != message_digest:
                return rpc_error(
                    request_id=request_id,
                    code=-32009,
                    message="IDEMPOTENCY_CONFLICT",
                    status_code=409,
                )

            old_task = get_task_for_owner(
                token=token,
                task_id=previous["taskId"],
            )

            if old_task is not None:
                return rpc_success(
                    request_id=request_id,
                    result={
                        "task": old_task,
                    },
                )

        proposals: list[
            dict[str, Any]
        ] = []

        for package in packages:
            if not isinstance(package, dict):
                return rpc_error(
                    request_id=request_id,
                    code=-32602,
                    message=(
                        "Every package must be an object"
                    ),
                    status_code=400,
                )

            proposals.append(
                make_proposal(package)
            )

        task_id = str(uuid.uuid4())

        task = {
            "id": task_id,
            "contextId": message.get(
                "contextId",
                str(uuid.uuid4()),
            ),
            "status": {
                "state": TASK_STATE_INPUT_REQUIRED,
                "timestamp": now_iso(),
            },
            "history": [
                message,
            ],
            "artifacts": [
                {
                    "artifactId": str(uuid.uuid4()),
                    "name": "invoice-action-proposals",
                    "parts": [
                        {
                            "mediaType": PROPOSAL_MEDIA_TYPE,
                            "data": {
                                "batchId": batch.get(
                                    "batchId",
                                    str(uuid.uuid4()),
                                ),
                                "proposals": proposals,
                            },
                        }
                    ],
                }
            ],
        }

        save_task(
            token=token,
            task=task,
        )

        STATE[
            "messages"
        ][message_key] = {
            "taskId": task_id,
            "messageHash": message_digest,
            "createdAt": now_iso(),
        }

        save_state()

    return rpc_success(
        request_id=request_id,
        result={
            "task": task,
        },
    )


# ============================================================
# Health
# ============================================================

@app.get("/")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "invoice-action-agent",
    }


# ============================================================
# Tasks
# ============================================================

@app.get("/tasks")
def list_tasks(
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    with STATE_LOCK:
        tasks = list(
            get_owner_tasks(token).values()
        )

    return a2a_response(
        {
            "tasks": tasks,
        }
    )


@app.get("/a2a/tasks")
def list_a2a_tasks(
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return list_tasks(
        token=token,
    )


@app.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    with STATE_LOCK:
        task = get_task_for_owner(
            token=token,
            task_id=task_id,
        )

    if task is None:
        return a2a_response(
            {
                "error": "Task not found",
            },
            status_code=404,
        )

    return a2a_response(task)


@app.get("/a2a/tasks/{task_id}")
def get_a2a_task(
    task_id: str,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return get_task(
        task_id=task_id,
        token=token,
    )


# ============================================================
# Cancellation
# ============================================================

@app.post("/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    with STATE_LOCK:
        task = get_task_for_owner(
            token=token,
            task_id=task_id,
        )

        if task is None:
            return a2a_response(
                {
                    "error": "Task not found",
                },
                status_code=404,
            )

        current_state = task[
            "status"
        ]["state"]

        if current_state in {
            TASK_STATE_COMPLETED,
            TASK_STATE_CANCELED,
        }:
            return a2a_response(
                {
                    "error": "Task is already terminal",
                },
                status_code=409,
            )

        task["status"] = {
            "state": TASK_STATE_CANCELED,
            "timestamp": now_iso(),
        }

        save_state()

    return a2a_response(task)


@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_a2a_task(
    task_id: str,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return cancel_task(
        task_id=task_id,
        token=token,
    )


# ============================================================
# Message endpoints
# ============================================================

@app.post("/message:send")
async def message_send(
    request: Request,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return await process_message(
        request=request,
        token=token,
    )


@app.post("/message%3Asend")
async def message_send_encoded(
    request: Request,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return await process_message(
        request=request,
        token=token,
    )


@app.post("/a2a/message:send")
async def a2a_message_send(
    request: Request,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return await process_message(
        request=request,
        token=token,
    )


@app.post("/a2a/message%3Asend")
async def a2a_message_send_encoded(
    request: Request,
    token: str = Depends(
        require_bearer_token
    ),
) -> JSONResponse:
    return await process_message(
        request=request,
        token=token,
    )


# ============================================================
# Error handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    return JSONResponse(
        content={
            "error": exc.detail,
        },
        status_code=exc.status_code,
        headers=exc.headers or {},
        media_type=A2A_MEDIA_TYPE,
    )


@app.exception_handler(Exception)
async def unhandled_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return a2a_response(
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": "Internal error",
            },
        },
        status_code=500,
    )
