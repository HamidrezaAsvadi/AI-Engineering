import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError
from pydantic import BaseModel, Field, field_validator
from .config import Settings
from .provider import OpenAIProvider
from .rag import RagService

log = logging.getLogger(__name__)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(None, ge=1, le=20)
    source: str | None = Field(None, max_length=160)

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Question must not be blank')
        return value.strip()

class DeleteDocument(BaseModel):
    source: str = Field(min_length=1, max_length=160)


def create_app(settings: Settings | None = None, service=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.rag = service if service is not None else RagService(settings, OpenAIProvider(settings))
        yield

    app = FastAPI(title='Chroma RAG Knowledge Assistant', version='1.0.0', lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)):
        key = settings.app_api_key.get_secret_value()
        if key and not secrets.compare_digest(authorization or '', f'Bearer {key}'):
            raise HTTPException(status_code=401, detail='Invalid or missing API token')

    def rag(request: Request):
        return request.app.state.rag

    @app.exception_handler(ValueError)
    async def bad_input(request, exc):
        return JSONResponse(status_code=400, content={'detail': str(exc)})

    @app.exception_handler(APIError)
    async def provider_error(request, exc):
        # Do not echo provider errors, credentials, or document content.
        log.warning('Provider request failed: %s', type(exc).__name__)
        return JSONResponse(status_code=502, content={'detail': 'Model provider request failed. Check key, model access, quota, and network.'})

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        log.error('Request failed: %s', type(exc).__name__)
        return JSONResponse(status_code=500, content={'detail': 'Request failed. Check server configuration and retry.'})

    @app.get('/health')
    def health():
        return {'status': 'ok'}  # Process liveness, not a provider availability check.

    @app.get('/api/documents', dependencies=[Depends(authorize)])
    def documents(service=Depends(rag)):
        return service.list_documents()

    @app.post('/api/documents', dependencies=[Depends(authorize)])
    def upload(file: UploadFile = File(...), service=Depends(rag)):
        try:
            raw = file.file.read(settings.max_upload_bytes + 1)
            if len(raw) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail='Maximum file size is 10 MiB')
            return service.ingest(raw, file.filename or '')
        finally:
            file.file.close()

    @app.delete('/api/documents', dependencies=[Depends(authorize)])
    def delete(body: DeleteDocument, service=Depends(rag)):
        return service.delete(body.source)

    @app.post('/api/search', dependencies=[Depends(authorize)])
    def search(body: Question, service=Depends(rag)):
        return {'sources': service.retrieve(body.question, body.top_k, body.source)}

    @app.post('/api/chat', dependencies=[Depends(authorize)])
    def chat(body: Question, service=Depends(rag)):
        return service.ask(body.question, body.top_k, body.source)

    static = Path(__file__).parent / 'static'
    app.mount('/static', StaticFiles(directory=static), name='static')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(static / 'index.html')

    return app

app = create_app()
