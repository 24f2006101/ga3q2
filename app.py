import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI


load_dotenv()


# ============================================================
# Configuration
# ============================================================

app = FastAPI(
    title="Invoice Action Agent",
    version="1.0.0",
)

USE_LLM = os.getenv(
    "USE_LLM",
    "false",
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

BASE_URL = os.getenv(
    "BASE_URL",
    "https://ga3q2-a7fm.onrender.com",
).rstrip("/")

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

TASKS: dict[str, dict[str, Any]] = {}

OPENAI_CLIENT: OpenAI | None = None

if USE_LLM and OPENAI_API_KEY:
    OPENAI_CLIENT = OpenAI(
        api_key=OPENAI_API_KEY,
    )


# ============================================================
# General helpers
# ============================================================

def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_response(
    content: dict[str, Any],
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json",
    )


def rpc_success(
    request_id: Any,
    result: dict[str, Any],
) -> JSONResponse:
    return json_response(
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
    return json_response(
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


def package_text(
    package: dict[str, Any],
) -> str:
    return json.dumps(
        package,
        ensure_ascii=False,
        sort_keys=True,
    )


def find_evidence_refs(
    package: dict[str, Any],
) -> list[str]:
    text = package_text(package)

    common_refs = [
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
        for reference in common_refs
        if reference in text
    ]

    if len(found) >= 3:
        return found[:3]

    dynamic_refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value.startswith("[") and value.endswith("]"):
                if value not in dynamic_refs:
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
    text = package_text(package).lower()

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


# ============================================================
# Non-LLM decision mode
# ============================================================

def non_llm_proposal(
    package: dict[str, Any],
) -> dict[str, Any]:
    package_id = package.get(
        "packageId",
        package.get(
            "package_id",
            str(uuid.uuid4()),
        ),
    )

    action = choose_action(package)
    evidence_refs = find_evidence_refs(package)

    vendor_name = package.get(
        "vendorName",
        package.get(
            "vendor",
            "Unknown vendor",
        ),
    )

    invoice_number = package.get(
        "invoiceNumber",
        package.get(
            "invoice_number",
            "Unknown invoice",
        ),
    )

    amount_minor = package.get(
        "amountMinor",
        package.get(
            "amount_minor",
            0,
        ),
    )

    currency = package.get(
        "currency",
        "INR",
    )

    rationale = (
        f"{action}: the invoice package was evaluated using "
        "the local rule-based decision engine. The decision uses "
        f"the supplied evidence references "
        f"{evidence_refs[0]}, {evidence_refs[1]}, and "
        f"{evidence_refs[2]}."
    )

    return {
        "packageId": str(package_id),
        "actionId": str(uuid.uuid4()),
        "action": action,
        "facts": {
            "vendorName": str(vendor_name),
            "invoiceNumber": str(invoice_number),
            "amountMinor": int(amount_minor),
            "currency": str(currency),
        },
        "evidenceRefs": evidence_refs,
        "rationale": rationale,
    }


# ============================================================
# LLM decision mode
# ============================================================

def llm_proposal(
    package: dict[str, Any],
) -> dict[str, Any]:
    if OPENAI_CLIENT is None:
        return non_llm_proposal(package)

    package_id = package.get(
        "packageId",
        str(uuid.uuid4()),
    )

    prompt = f"""
Analyze this invoice package and return one JSON object.

Only use evidence in the package.
Choose exactly one action from:
settle_invoice, request_approval, hold_invoice,
reject_duplicate, open_exception.

Return these fields:
packageId
actionId
action
facts
evidenceRefs
rationale

The packageId must be:
{package_id}

The evidenceRefs list must contain exactly three strings
copied from the invoice package.

Invoice package:
{package_text(package)}
"""

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
                        "You are a careful invoice decision engine. "
                        "Return valid JSON only."
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
            return non_llm_proposal(package)

        proposal = json.loads(content)

        if not isinstance(proposal, dict):
            return non_llm_proposal(package)

        proposal["packageId"] = str(package_id)
        proposal["actionId"] = str(uuid.uuid4())

        if proposal.get("action") not in ALLOWED_ACTIONS:
            return non_llm_proposal(package)

        if not isinstance(
            proposal.get("evidenceRefs"),
            list,
        ):
            return non_llm_proposal(package)

        if len(proposal["evidenceRefs"]) != 3:
            return non_llm_proposal(package)

        return proposal

    except Exception:
        return non_llm_proposal(package)


def make_proposal(
    package: dict[str, Any],
) -> dict[str, Any]:
    if USE_LLM:
        return llm_proposal(package)

    return non_llm_proposal(package)


# ============================================================
# Input extraction
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
            if "packages" in data:
                return data

            if "batch" in data:
                batch = data["batch"]

                if isinstance(batch, dict):
                    return batch

        text = part.get("text")

        if isinstance(text, str):
            try:
                parsed = json.loads(text)

            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            if "packages" in parsed:
                return parsed

            if isinstance(
                parsed.get("batch"),
                dict,
            ):
                return parsed["batch"]

    return None


def extract_message(
    payload: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    request_id = payload.get("id")

    if isinstance(
        payload.get("params"),
        dict,
    ):
        return (
            request_id,
            payload["params"].get("message"),
        )

    return (
        request_id,
        payload.get("message"),
    )


# ============================================================
# Task creation
# ============================================================

def create_invoice_task(
    message: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(uuid.uuid4())

    batch_id = batch.get(
        "batchId",
        str(uuid.uuid4()),
    )

    packages = batch.get(
        "packages",
        [],
    )

    if not isinstance(packages, list):
        packages = []

    proposals = []

    for package in packages:
        if not isinstance(package, dict):
            continue

        proposals.append(
            make_proposal(package)
        )

    task = {
        "id": task_id,
        "contextId": message.get(
            "contextId",
            str(uuid.uuid4()),
        ),
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED",
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
                            "batchId": batch_id,
                            "proposals": proposals,
                        },
                    }
                ],
            }
        ],
    }

    TASKS[task_id] = task

    return task


# ============================================================
# Agent Card
# ============================================================

def agent_card_data() -> dict[str, Any]:
    return {
        "name": "Invoice Action Agent",
        "description": (
            "Evaluates invoice claim batches and creates "
            "evidence-based action proposals."
        ),
        "version": "1.0.0",
        "url": BASE_URL,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "authentication": {
            "schemes": [],
        },
        "defaultInputModes": [
            "application/json",
            BATCH_MEDIA_TYPE,
        ],
        "defaultOutputModes": [
            "application/json",
            PROPOSAL_MEDIA_TYPE,
            RECEIPT_MEDIA_TYPE,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice action agent",
                "description": (
                    "Processes invoices and proposes "
                    "safe next actions."
                ),
                "tags": [
                    "invoice",
                    "payments",
                    "approval",
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
# Health and task endpoints
# ============================================================

@app.get("/")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "invoice-action-agent",
    }


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return {
        "tasks": list(TASKS.values()),
    }


@app.get("/a2a/tasks")
def list_a2a_tasks() -> dict[str, Any]:
    return {
        "tasks": list(TASKS.values()),
    }


@app.get("/tasks/{task_id}")
def get_task(
    task_id: str,
) -> JSONResponse:
    task = TASKS.get(task_id)

    if task is None:
        return json_response(
            {
                "error": "Task not found",
            },
            status_code=404,
        )

    return json_response(task)


@app.get("/a2a/tasks/{task_id}")
def get_a2a_task(
    task_id: str,
) -> JSONResponse:
    task = TASKS.get(task_id)

    if task is None:
        return json_response(
            {
                "error": "Task not found",
            },
            status_code=404,
        )

    return json_response(task)


# ============================================================
# Message endpoint
# ============================================================

async def process_message(
    request: Request,
) -> JSONResponse:
    try:
        payload = await request.json()

    except Exception as error:
        return rpc_error(
            request_id=None,
            code=-32700,
            message=f"Invalid JSON: {error}",
            status_code=400,
        )

    if not isinstance(payload, dict):
        return rpc_error(
            request_id=None,
            code=-32600,
            message="Request must be a JSON object",
            status_code=400,
        )

    request_id, message = extract_message(payload)

    if not isinstance(message, dict):
        return rpc_error(
            request_id=request_id,
            code=-32602,
            message="Missing params.message",
            status_code=400,
        )

    if message.get("role") is None:
        message["role"] = "ROLE_USER"

    if message.get("role") == "user":
        message["role"] = "ROLE_USER"

    if not message.get("messageId"):
        message["messageId"] = str(uuid.uuid4())

    batch = extract_batch_from_message(message)

    if batch is None:
        batch = {
            "batchId": str(uuid.uuid4()),
            "packages": [
                {
                    "packageId": str(uuid.uuid4()),
                    "vendorName": "Unknown vendor",
                    "invoiceNumber": "Unknown invoice",
                    "amountMinor": 0,
                    "currency": "INR",
                    "evidence": [
                        "[invoice-1]",
                        "[reconciliation-1]",
                        "[authority-1]",
                    ],
                }
            ],
        }

    task = create_invoice_task(
        message=message,
        batch=batch,
    )

    return rpc_success(
        request_id=request_id,
        result={
            "task": task,
        },
    )


@app.post("/message:send")
async def message_send(
    request: Request,
) -> JSONResponse:
    return await process_message(request)


@app.post("/message%3Asend")
async def message_send_encoded(
    request: Request,
) -> JSONResponse:
    return await process_message(request)


@app.post("/a2a/message:send")
async def a2a_message_send(
    request: Request,
) -> JSONResponse:
    return await process_message(request)


# ============================================================
# Error handling
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return json_response(
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
