"""HTTP API.  Run from the project root:

    uvicorn app.api.server:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""
import os
import jwt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr

from app.ai.assistant import answer_question
from app.api.uploads import MAX_UPLOAD_BYTES, UploadError, parse_upload
from app.database.connection import get_ingest_connection, get_query_connection
from app.database.ingest import delete_dataset, get_user_datasets, ingest_dataframe, safe_identifiers, rename_dataset
from app.database.tenancy import schema_name
from app.api.dashboard import get_dashboard_data, get_variable_models
from app.database.auth import ensure_auth_tables, create_user, get_user_by_email, verify_password, get_user_by_id, mark_user_verified
from fastapi.responses import FileResponse

app = FastAPI(title="AI Data Analyst")

# Security configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "super-secret-development-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@app.on_event("startup")
def on_startup():
    ensure_auth_tables()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Validates JWT and returns the user object.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = get_user_by_id(user_id)
    if not user:
        raise credentials_exception

    # Note: verification is required for login/use by the issue description.
    if not user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified"
        )

    if not user.get("cohere_api_key"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cohere API key missing"
        )

    try:
        schema_name(user_id)  # validates the format for multi-tenancy
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return user


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    cohere_api_key: str = Field(min_length=10)


class Token(BaseModel):
    access_token: str
    token_type: str


@app.post("/auth/register", status_code=201)
def register(user_in: UserRegister):
    existing = get_user_by_email(user_in.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = create_user(user_in.email, user_in.password, user_in.cohere_api_key)
    return {"id": user["id"], "email": user["email"], "is_verified": user["is_verified"]}


@app.post("/auth/login", response_model=Token)
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["id"])}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/verify")
def verify_email(email: str):
    """
    Mock verification endpoint. In a real app, this would be triggered by an email link.
    """
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    mark_user_verified(str(user["id"]))
    return {"message": "Email verified successfully"}


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
        user: Annotated[dict, Depends(current_user)],
        file: Annotated[UploadFile, File()],
        table_name: Annotated[str | None, Form()] = None,
        has_header: Annotated[bool, Form()] = True,
        replace: Annotated[bool, Form()] = False,
):
    """Upload a CSV/TSV, clean it, and load it into the caller's own schema."""
    user_id = str(user["id"])
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
        user: Annotated[dict, Depends(current_user)],
        include_profile: bool = False,
):
    """The caller's tables, with column types. Add ?include_profile=true for full stats."""
    user_id = str(user["id"])
    with get_ingest_connection() as conn:
        datasets = get_user_datasets(conn, user_id)
    if not include_profile:
        for dataset in datasets:
            dataset.pop("profile", None)
    return datasets


@app.delete("/datasets/{table_name}", status_code=204)
def remove_dataset(
        table_name: str,
        user: Annotated[dict, Depends(current_user)],
):
    """Permanently delete a dataset."""
    user_id = str(user["id"])
    with get_ingest_connection() as conn:
        # Check existence first
        datasets = get_user_datasets(conn, user_id)
        if not any(d["table_name"] == table_name for d in datasets):
            raise HTTPException(status_code=404, detail=f"Dataset '{table_name}' not found.")

        delete_dataset(conn, user_id, table_name)
    return


class RenameRequest(BaseModel):
    new_name: str = Field(min_length=1, max_length=63)


@app.patch("/datasets/{table_name}", status_code=200)
def rename_user_dataset(
        table_name: str,
        request: RenameRequest,
        user: Annotated[dict, Depends(current_user)],
):
    """Rename a dataset."""
    user_id = str(user["id"])
    with get_ingest_connection() as conn:
        # Check existence first
        datasets = get_user_datasets(conn, user_id)
        if not any(d["table_name"] == table_name for d in datasets):
            raise HTTPException(status_code=404, detail=f"Dataset '{table_name}' not found.")

        try:
            rename_dataset(conn, user_id, table_name, request.new_name)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))

    return {"message": "Dataset renamed successfully", "new_name": request.new_name}


@app.get("/datasets/{table_name}/dashboard")
def get_dataset_dashboard(
        table_name: str,
        user: Annotated[dict, Depends(current_user)],
):
    """Compute detailed dashboard statistics for a dataset."""
    user_id = str(user["id"])
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


@app.get("/datasets/{table_name}/variables/{column}/models")
def get_variable_models_endpoint(
        table_name: str,
        column: str,
        user: Annotated[dict, Depends(current_user)],
):
    """Simple linear or logistic regressions with one column as the outcome."""
    user_id = str(user["id"])
    with get_ingest_connection() as conn:
        datasets = get_user_datasets(conn, user_id)
        dataset = next((d for d in datasets if d["table_name"] == table_name), None)

    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{table_name}' not found.")
    if column not in dataset["profile"]["columns"]:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found in '{table_name}'.")

    with get_query_connection() as query_conn:
        try:
            return get_variable_models(query_conn, user_id, table_name, dataset["profile"], column)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
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
def ask(request: AskRequest, user: Annotated[dict, Depends(current_user)]):
    """Ask a plain-English question about the caller's data."""
    user_id = str(user["id"])
    api_key = user.get("cohere_api_key")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="The question is empty.")

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
            import cohere
            client = cohere.ClientV2(api_key=api_key)
            return answer_question(question, user_id, datasets, query_conn, client=client, history=history)
        except Exception as error:  # e.g. AI provider rate limit
            raise HTTPException(status_code=503, detail=str(error))