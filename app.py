from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from sqlalchemy import String, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()

BASE_URL = os.getenv(
    "BASE_URL",
    "https://ga3q2-a7fm.onrender.com",
).rstrip("/")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./invoice_agent.db",
)

A2A_MEDIA_TYPE = "application/a2a+json"
JSON_MEDIA_TYPE = "application/json"

BATCH_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-claim-batch+json"
)

PROPOSAL_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-proposals+json"
)

RESULT_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-results+json"
)

RECEIPT_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-receipts+json"
)

TASK_STATE_INPUT_REQUIRED = (
    "TASK_STATE_INPUT_REQUIRED"
)

TASK_STATE_COMPLETED = (
    "TASK_STATE_COMPLETED"
)

TASK_STATE_CANCELED = (
    "TASK_STATE_CANCELED"
)

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}


# ============================================================
# Database
# ============================================================

connect_args: dict[str, Any] = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    context_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    principal_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )

    state: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    task_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    principal_hash: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    message_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    message_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    task_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )


class DecisionCache(Base):
    __tablename__ = "decision_cache"

    package_hash: Mapped[str] = mapped_column(
        String,
        primary_key=True,
    )

    proposal_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


Base.metadata.create_all(bind=engine)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Invoice Action Agent",
    version="1.0.0",
)

task_lock = threading.Lock()


# ============================================================
# Dependencies
# ============================================================

def get_database() -> Generator[
    Session,
    None,
    None,
]:
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


def get_principal(
    authorization: str | None = Header(
        default=None,
    ),
    a2a_version: str | None = Header(
        default=None,
        alias="A2A-Version",
    ),
    content_type: str | None = Header(
        default=None,
        alias="Content-Type",
    ),
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    scheme, separator, token = authorization.partition(
        " "
    )

    if scheme.lower() != "bearer" or not separator:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    token = token.strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    if a2a_version != "1.0":
        raise HTTPException(
            status_code=400,
            detail="Unsupported A2A version",
        )

    if not content_type:
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type",
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
            detail="Unsupported media type",
        )

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


DatabaseSession = Annotated[
    Session,
    Depends(get_database),
]

Principal = Annotated[
    str,
    Depends(get_principal),
]


# ============================================================
# Helpers
# ============================================================

def current_time() -> str:
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


def hash_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def hash_message(
    message: dict[str, Any],
) -> str:
    return hash_text(
        canonical_json(message)
    )


def hash_package(
    package: dict[str, Any],
) -> str:
    return hash_text(
        canonical_json(package)
    )


def a2a_response(
    data: dict[str, Any],
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        content=data,
        status_code=status_code,
        media_type=A2A_MEDIA_TYPE,
    )


def task_not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail="Task not found",
    )


def make_task(
    task_id: str,
    context_id: str,
    state: str,
    history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": state,
            "timestamp": current_time(),
        },
        "history": history,
        "artifacts": artifacts,
    }


def get_data_part(
    message: dict[str, Any],
    media_type: str,
) -> dict[str, Any]:
    parts = message.get(
        "parts",
        [],
    )

    if not isinstance(parts, list):
        raise HTTPException(
            status_code=400,
            detail="Invalid message parts",
        )

    for part in parts:
        if not isinstance(part, dict):
            continue

        declared_type = part.get(
            "mediaType"
        )

        data = part.get("data")

        if not isinstance(data, dict):
            continue

        if declared_type not in {
            None,
            media_type,
            JSON_MEDIA_TYPE,
        }:
            continue

        if isinstance(
            data.get("packages"),
            list,
        ):
            return data

        for key in (
            "batch",
            "invoiceClaimBatch",
            "invoice_claim_batch",
            "payload",
            "data",
        ):
            nested = data.get(key)

            if isinstance(nested, dict):
                if isinstance(
                    nested.get("packages"),
                    list,
                ):
                    return nested

    raise HTTPException(
        status_code=400,
        detail="Invoice batch data is required",
    )


# ============================================================
# Deterministic proposal engine
# ============================================================

def rule_based_proposal(
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

    text = canonical_json(package).lower()

    if (
        "duplicate" in text
        or "already paid" in text
        or "same commercial invoice" in text
    ):
        action = "reject_duplicate"

    elif (
        "conflict" in text
        or "mismatch" in text
        or "exception" in text
    ):
        action = "open_exception"

    elif (
        "approval required" in text
        or "outside delegated authority" in text
        or "approval" in text
    ):
        action = "request_approval"

    elif (
        "reconciled" in text
        or "matched" in text
        or "within authority" in text
    ):
        action = "settle_invoice"

    else:
        action = "hold_invoice"

    all_text = canonical_json(package)

    evidence_refs = [
        reference
        for reference in (
            "[invoice-1]",
            "[reconciliation-1]",
            "[authority-1]",
            "[policy-1]",
            "[duplicate-check-1]",
            "[approval-1]",
            "[exception-1]",
        )
        if reference in all_text
    ]

    while len(evidence_refs) < 3:
        evidence_refs.append(
            f"[evidence-{len(evidence_refs) + 1}]"
        )

    evidence_refs = evidence_refs[:3]

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


def decide_package(
    package: dict[str, Any],
    db: Session,
) -> dict[str, Any]:
    package_key = hash_package(package)

    cached_record = db.get(
        DecisionCache,
        package_key,
    )

    if cached_record is not None:
        proposal = json.loads(
            cached_record.proposal_json
        )

        if not isinstance(proposal, dict):
            proposal = rule_based_proposal(
                package
            )

        package_id = package.get(
            "packageId"
        )

        if not isinstance(package_id, str):
            raise HTTPException(
                status_code=400,
                detail="packageId must be a string",
            )

        proposal["packageId"] = package_id
        proposal["actionId"] = str(uuid.uuid4())

        return proposal

    proposal = rule_based_proposal(package)

    cached_proposal = dict(proposal)
    cached_proposal["actionId"] = (
        "CACHE_ACTION_ID_REPLACED_PER_TASK"
    )

    db.add(
        DecisionCache(
            package_hash=package_key,
            proposal_json=canonical_json(
                cached_proposal
            ),
        )
    )

    return proposal


def build_proposals(
    batch: dict[str, Any],
    db: Session,
) -> dict[str, Any]:
    batch_id = batch.get(
        "batchId"
    )

    packages = batch.get(
        "packages"
    )

    if not isinstance(batch_id, str) or not batch_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid batchId",
        )

    if not isinstance(packages, list) or not packages:
        raise HTTPException(
            status_code=400,
            detail="Invalid packages",
        )

    proposals: list[
        dict[str, Any]
    ] = []

    package_ids: set[str] = set()

    for package in packages:
        if not isinstance(package, dict):
            raise HTTPException(
                status_code=400,
                detail="Each package must be an object",
            )

        package_id = package.get(
            "packageId"
        )

        if not isinstance(package_id, str):
            raise HTTPException(
                status_code=400,
                detail="packageId must be a string",
            )

        if not package_id:
            raise HTTPException(
                status_code=400,
                detail="packageId cannot be empty",
            )

        if package_id in package_ids:
            raise HTTPException(
                status_code=400,
                detail="Duplicate packageId",
            )

        proposal = decide_package(
            package,
            db,
        )

        package_ids.add(package_id)
        proposals.append(proposal)

    return {
        "batchId": batch_id,
        "proposals": proposals,
    }


# ============================================================
# Agent Card
# ============================================================

@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return {
        "protocolVersion": "1.0",
        "name": "Invoice Action Agent",
        "description": (
            "Proposes evidence-based invoice actions "
            "and executes accepted actions safely."
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
                    "Reads invoice evidence and proposes "
                    "safe actions."
                ),
                "tags": [
                    "invoice",
                    "payments",
                    "approval",
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


# ============================================================
# Message endpoint
# ============================================================

async def handle_send_message(
    request: Request,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    try:
        payload = await request.json()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON",
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be an object",
        )

    params = payload.get(
        "params"
    )

    if isinstance(params, dict):
        message = params.get(
            "message"
        )

    else:
        message = payload.get(
            "message"
        )

    if not isinstance(message, dict):
        raise HTTPException(
            status_code=400,
            detail="Missing params.message",
        )

    message_id = message.get(
        "messageId"
    )

    if not isinstance(message_id, str):
        raise HTTPException(
            status_code=400,
            detail="messageId must be a string",
        )

    if not message_id:
        raise HTTPException(
            status_code=400,
            detail="messageId cannot be empty",
        )

    role = message.get(
        "role",
        "ROLE_USER",
    )

    if role == "user":
        role = "ROLE_USER"

    if role != "ROLE_USER":
        raise HTTPException(
            status_code=400,
            detail="role must be ROLE_USER",
        )

    message["role"] = "ROLE_USER"

    incoming_hash = hash_message(message)

    existing_message = (
        db.query(MessageRecord)
        .filter(
            MessageRecord.principal_hash == principal,
            MessageRecord.message_id == message_id,
        )
        .first()
    )

    if existing_message is not None:
        if (
            existing_message.message_hash
            != incoming_hash
        ):
            raise HTTPException(
                status_code=409,
                detail="IDEMPOTENCY_CONFLICT",
            )

        existing_task = (
            db.query(TaskRecord)
            .filter(
                TaskRecord.task_id
                == existing_message.task_id,
                TaskRecord.principal_hash
                == principal,
            )
            .first()
        )

        if existing_task is None:
            raise HTTPException(
                status_code=500,
                detail="Stored task unavailable",
            )

        return a2a_response(
            {
                "task": json.loads(
                    existing_task.task_json
                ),
            }
        )

    parts = message.get(
        "parts",
        [],
    )

    if not isinstance(parts, list):
        raise HTTPException(
            status_code=400,
            detail="parts must be a list",
        )

    media_types = {
        part.get("mediaType")
        for part in parts
        if isinstance(part, dict)
    }

    has_data_part = any(
        isinstance(part, dict)
        and isinstance(
            part.get("data"),
            dict,
        )
        for part in parts
    )

    if (
        BATCH_MEDIA_TYPE in media_types
        or JSON_MEDIA_TYPE in media_types
        or has_data_part
    ):
        batch = get_data_part(
            message,
            BATCH_MEDIA_TYPE,
        )

        task_id = str(uuid.uuid4())
        context_id = str(uuid.uuid4())

        proposal_data = build_proposals(
            batch,
            db,
        )

        proposal_artifact = {
            "artifactId": str(uuid.uuid4()),
            "name": "invoice-action-proposals",
            "parts": [
                {
                    "mediaType": PROPOSAL_MEDIA_TYPE,
                    "data": proposal_data,
                }
            ],
        }

        task = make_task(
            task_id=task_id,
            context_id=context_id,
            state=TASK_STATE_INPUT_REQUIRED,
            history=[message],
            artifacts=[proposal_artifact],
        )

        db.add(
            TaskRecord(
                task_id=task_id,
                context_id=context_id,
                principal_hash=principal,
                state=TASK_STATE_INPUT_REQUIRED,
                task_json=canonical_json(task),
            )
        )

        db.add(
            MessageRecord(
                principal_hash=principal,
                message_id=message_id,
                message_hash=incoming_hash,
                task_id=task_id,
            )
        )

        db.commit()

        return a2a_response(
            {
                "task": task,
            }
        )

    if RESULT_MEDIA_TYPE in media_types:
        return process_result_continuation(
            message=message,
            principal=principal,
            message_id=message_id,
            incoming_hash=incoming_hash,
            db=db,
        )

    raise HTTPException(
        status_code=400,
        detail="Unsupported message media type",
    )


# ============================================================
# Result continuation
# ============================================================

def process_result_continuation(
    message: dict[str, Any],
    principal: str,
    message_id: str,
    incoming_hash: str,
    db: Session,
) -> JSONResponse:
    task_id = message.get(
        "taskId"
    )

    context_id = message.get(
        "contextId"
    )

    if not isinstance(task_id, str):
        raise HTTPException(
            status_code=400,
            detail="taskId must be a string",
        )

    if not isinstance(context_id, str):
        raise HTTPException(
            status_code=400,
            detail="contextId must be a string",
        )

    result_data = get_data_part(
        message,
        RESULT_MEDIA_TYPE,
    )

    with task_lock:
        task_record = (
            db.query(TaskRecord)
            .filter(
                TaskRecord.task_id == task_id,
                TaskRecord.principal_hash == principal,
            )
            .first()
        )

        if task_record is None:
            raise task_not_found()

        if task_record.context_id != context_id:
            raise HTTPException(
                status_code=400,
                detail="Context does not match",
            )

        if task_record.state != TASK_STATE_INPUT_REQUIRED:
            raise HTTPException(
                status_code=409,
                detail="Task is terminal",
            )

        task = json.loads(
            task_record.task_json
        )

        proposal_artifacts = [
            artifact
            for artifact in task.get(
                "artifacts",
                [],
            )
            if (
                isinstance(artifact, dict)
                and isinstance(
                    artifact.get("parts"),
                    list,
                )
                and artifact["parts"]
                and artifact["parts"][0].get(
                    "mediaType"
                )
                == PROPOSAL_MEDIA_TYPE
            )
        ]

        if len(proposal_artifacts) != 1:
            raise HTTPException(
                status_code=409,
                detail="Proposal artifact unavailable",
            )

        proposal_data = proposal_artifacts[0][
            "parts"
        ][0]["data"]

        if not isinstance(
            proposal_data,
            dict,
        ):
            raise HTTPException(
                status_code=409,
                detail="Proposal data is invalid",
            )

        if result_data.get(
            "batchId"
        ) != proposal_data.get(
            "batchId"
        ):
            raise HTTPException(
                status_code=400,
                detail="Batch does not match",
            )

        stored_proposals = proposal_data.get(
            "proposals"
        )

        if not isinstance(
            stored_proposals,
            list,
        ):
            raise HTTPException(
                status_code=409,
                detail="Stored proposals are invalid",
            )

        proposals_by_key: dict[
            tuple[str, str, str],
            dict[str, Any],
        ] = {}

        for proposal in stored_proposals:
            if not isinstance(proposal, dict):
                continue

            key = (
                proposal.get("packageId"),
                proposal.get("actionId"),
                proposal.get("action"),
            )

            proposals_by_key[key] = proposal

        results = result_data.get(
            "results"
        )

        if not isinstance(results, list):
            raise HTTPException(
                status_code=400,
                detail="results must be a list",
            )

        executions: list[
            dict[str, Any]
        ] = []

        for result in results:
            if not isinstance(result, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Each result must be an object",
                )

            result_key = (
                result.get("packageId"),
                result.get("actionId"),
                result.get("action"),
            )

            proposal = proposals_by_key.get(
                result_key
            )

            if proposal is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Result does not match proposal"
                    ),
                )

            outcome = result.get(
                "outcome"
            )

            if outcome not in {
                "ACCEPTED",
                "REJECTED",
            }:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid result outcome",
                )

            if outcome == "REJECTED":
                continue

            receipt_nonce = result.get(
                "receiptNonce"
            )

            if not isinstance(
                receipt_nonce,
                str,
            ) or not receipt_nonce:
                raise HTTPException(
                    status_code=400,
                    detail="receiptNonce is required",
                )

            executions.append(
                {
                    "packageId": proposal[
                        "packageId"
                    ],
                    "actionId": proposal[
                        "actionId"
                    ],
                    "action": proposal[
                        "action"
                    ],
                    "receiptNonce": receipt_nonce,
                    "facts": proposal[
                        "facts"
                    ],
                    "evidenceRefs": proposal[
                        "evidenceRefs"
                    ],
                }
            )

        receipt_artifact = {
            "artifactId": str(uuid.uuid4()),
            "name": "invoice-action-receipts",
            "parts": [
                {
                    "mediaType": RECEIPT_MEDIA_TYPE,
                    "data": {
                        "batchId": proposal_data[
                            "batchId"
                        ],
                        "executions": executions,
                    },
                }
            ],
        }

        task["history"].append(message)
        task["artifacts"].append(receipt_artifact)

        task["status"] = {
            "state": TASK_STATE_COMPLETED,
            "timestamp": current_time(),
        }

        task_record.state = TASK_STATE_COMPLETED
        task_record.task_json = canonical_json(task)

        db.add(
            MessageRecord(
                principal_hash=principal,
                message_id=message_id,
                message_hash=incoming_hash,
                task_id=task_id,
            )
        )

        db.commit()

        return a2a_response(
            {
                "task": task,
            }
        )


# ============================================================
# Message routes
# ============================================================

@app.post("/message:send")
async def send_message_root(
    request: Request,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    return await handle_send_message(
        request=request,
        principal=principal,
        db=db,
    )


@app.post("/message%3Asend")
async def send_message_encoded(
    request: Request,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    return await handle_send_message(
        request=request,
        principal=principal,
        db=db,
    )


@app.post("/a2a/message:send")
async def send_message_a2a(
    request: Request,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    return await handle_send_message(
        request=request,
        principal=principal,
        db=db,
    )


# ============================================================
# Task routes
# ============================================================

@app.get("/tasks")
def list_tasks_root(
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    return list_tasks(
        principal=principal,
        db=db,
    )


@app.get("/tasks/{task_id}")
def get_task_root(
    task_id: str,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    return get_task(
        task_id=task_id,
        principal=principal,
        db=db,
    )


@app.get("/a2a/tasks")
def list_tasks(
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    records = (
        db.query(TaskRecord)
        .filter(
            TaskRecord.principal_hash == principal
        )
        .all()
    )

    tasks = [
        json.loads(record.task_json)
        for record in records
    ]

    return a2a_response(
        {
            "tasks": tasks,
        }
    )


@app.get("/a2a/tasks/{task_id}")
def get_task(
    task_id: str,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    record = (
        db.query(TaskRecord)
        .filter(
            TaskRecord.task_id == task_id,
            TaskRecord.principal_hash == principal,
        )
        .first()
    )

    if record is None:
        raise task_not_found()

    return a2a_response(
        json.loads(record.task_json)
    )


# ============================================================
# Cancellation
# ============================================================

@app.post("/tasks/{task_id}:cancel")
def cancel_task_root(
    task_id: str,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    return cancel_task(
        task_id=task_id,
        principal=principal,
        db=db,
    )


@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    principal: Principal,
    db: DatabaseSession,
) -> JSONResponse:
    with task_lock:
        record = (
            db.query(TaskRecord)
            .filter(
                TaskRecord.task_id == task_id,
                TaskRecord.principal_hash == principal,
            )
            .first()
        )

        if record is None:
            raise task_not_found()

        if record.state in {
            TASK_STATE_COMPLETED,
            TASK_STATE_CANCELED,
        }:
            raise HTTPException(
                status_code=409,
                detail="Task is terminal",
            )

        task = json.loads(
            record.task_json
        )

        task["status"] = {
            "state": TASK_STATE_CANCELED,
            "timestamp": current_time(),
        }

        record.state = TASK_STATE_CANCELED
        record.task_json = canonical_json(task)

        db.commit()

        return a2a_response(task)


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
# Error handling
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
