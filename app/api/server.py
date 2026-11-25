"""HTTP API.  Run from the project root:

    uvicorn app.api.server:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai.assistant import answer_question
from app.api.uploads import MAX_UPLOAD_BYTES, UploadError, parse_upload
from app.database.connection import get_ingest_connection, get_query_connection
from app.database.ingest import delete_dataset, get_user_datasets, ingest_dataframe, safe_identifiers
from app.database.tenancy import schema_name
from app.api.dashboard import get_dashboard_data
from fastapi.responses import FileResponse

app = FastAPI(title="AI Data Analyst")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


def current_user(x_user_id: Annotated[str, Header()]):
    """
    DEVELOPMENT ONLY: trusts whatever X-User-Id the caller sends. Anyone who can
    reach this server can read any user's data. Replace with real authentication
    before exposing it beyond your own machine.
    """
    try:
        schema_name(x_user_id)  # validates the format
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return x_user_id


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}

INDEX_PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_PAGE)


@app.post("/datasets", status_code=201)
def upload_dataset(
        user_id: Annotated[str, Depends(current_user)],
        file: Annotated[UploadFile, File()],
        table_name: Annotated[str | None, Form()] = None,
        has_header: Annotated[bool, Form()] = True,
        replace: Annotated[bool, Form()] = False,
):
    """Upload a CSV/TSV, clean it, and load it into the caller's own schema."""
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        df = parse_upload(content, has_header=has_header)
    except UploadError as error:
        raise HTTPException(status_code=400, detail=str(error))

    name = safe_identifiers([table_name or Path(file.filename or "dataset").stem])[0]

    with get_ingest_connection() as conn:
        existing = {d["table_name"] for d in get_user_datasets(conn, user_id)}
        if name in existing and not replace:
            raise HTTPException(
                status_code=409,
                detail=f"Dataset '{name}' already exists. Send replace=true to overwrite it.",
            )
        return ingest_dataframe(
            conn, df, user_id, name,
            original_filename=file.filename,
            if_exists="replace" if replace else "fail",
        )


@app.get("/datasets")
def list_datasets(
        user_id: Annotated[str, Depends(current_user)],
        include_profile: bool = False,
):
    """The caller's tables, with column types. Add ?include_profile=true for full stats."""
    with get_ingest_connection() as conn:
        datasets = get_user_datasets(conn, user_id)
    if not include_profile:
        for dataset in datasets:
            dataset.pop("profile", None)
    return datasets


@app.delete("/datasets/{table_name}", status_code=204)
def remove_dataset(
        table_name: str,
        user_id: Annotated[str, Depends(current_user)],
):
    """Permanently delete a dataset."""
    with get_ingest_connection() as conn:
        # Check existence first
        datasets = get_user_datasets(conn, user_id)
        if not any(d["table_name"] == table_name for d in datasets):
            raise HTTPException(status_code=404, detail=f"Dataset '{table_name}' not found.")

        delete_dataset(conn, user_id, table_name)
    return


@app.get("/datasets/{table_name}/dashboard")
def get_dataset_dashboard(
        table_name: str,
        user_id: Annotated[str, Depends(current_user)],
):
    """Compute detailed dashboard statistics for a dataset."""
    with get_ingest_connection() as conn:
        datasets = get_user_datasets(conn, user_id)
        dataset = next((d for d in datasets if d["table_name"] == table_name), None)
    
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{table_name}' not found.")
    
    with get_query_connection() as query_conn:
        try:
            return get_dashboard_data(query_conn, user_id, table_name, dataset["profile"])
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error))


class Turn(BaseModel):
    """One earlier exchange. Send back the "turn" object each /ask response returns."""
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=8000)
    sql: list[str] = Field(default_factory=list, max_length=10)
    analyses: list[dict] = Field(default_factory=list, max_length=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    table: str | None = None  # limit the question to one of your datasets
    history: list[Turn] = Field(default_factory=list, max_length=20)

@app.post("/ask")
def ask(request: AskRequest, user_id: Annotated[str, Depends(current_user)]):
    """Ask a plain-English question about the caller's data."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="The question is empty.")
    if not os.environ.get("COHERE_API_KEY"):
        raise HTTPException(status_code=500, detail="The server has no COHERE_API_KEY configured.")

    with get_ingest_connection() as conn:
        datasets = get_user_datasets(conn, user_id)
    if not datasets:
        raise HTTPException(status_code=404, detail="No datasets yet. Upload one with POST /datasets.")
    if request.table:
        datasets = [d for d in datasets if d["table_name"] == request.table]
        if not datasets:
            raise HTTPException(status_code=404, detail=f"No dataset named '{request.table}'.")

    history = [turn.model_dump() for turn in request.history]
    with get_query_connection() as query_conn:
        try:
            return answer_question(question, user_id, datasets, query_conn, history=history)
        except RuntimeError as error:  # e.g. AI provider rate limit
            raise HTTPException(status_code=503, detail=str(error))