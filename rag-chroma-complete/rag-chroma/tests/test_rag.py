from hashlib import sha256
from io import BytesIO
import math
import re
import json
import httpx
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import SecretStr
from pypdf import PdfWriter
import pytest
from app.config import Settings
from app.documents import extract, source_name, split_pages
from app.main import create_app
from app.provider import OpenAIProvider
from app.rag import RagService, UNKNOWN
from evaluation.run import retrieval_metrics

class FakeProvider:
    """Lexical vectors test plumbing, NOT semantic retrieval quality."""
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.answer = 'The gateway buffers telemetry for 24 hours. [S1]'
    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError('Simulated provider outage')
        result = []
        for text in texts:
            vector = [0.] * 64
            for word in re.findall(r'\w+', text.lower()):
                vector[int(sha256(word.encode()).hexdigest(), 16) % 64] += 1
            norm = math.sqrt(sum(x*x for x in vector)) or 1
            result.append([x/norm for x in vector])
        return result
    def generate(self, question, sources):
        return self.answer

@pytest.fixture
def setup(tmp_path):
    config = Settings(_env_file=None, chroma_path=tmp_path/'chroma', openai_api_key='', app_api_key='')
    provider = FakeProvider()
    return config, provider, RagService(config, provider)

def test_empty(setup):
    _, provider, service = setup
    assert service.ask('Anything?')['answer'] == UNKNOWN
    assert provider.calls == 0

def test_idempotence_replacement(setup):
    _, provider, service = setup
    raw = b'Gateway buffers telemetry for 24 hours.'
    assert service.ingest(raw, 'gateway.md')['status'] == 'indexed'
    assert service.ingest(raw, 'gateway.md')['status'] == 'unchanged'
    assert provider.calls == 1
    service.ingest(b'Gateway buffers telemetry for 48 hours.', 'gateway.md')
    assert service.collection.get(include=['documents'])['documents'] == ['Gateway buffers telemetry for 48 hours.']

def test_provider_failure_preserves_old(setup):
    _, provider, service = setup
    service.ingest(b'Original document.', 'a.txt')
    provider.fail = True
    with pytest.raises(RuntimeError):
        service.ingest(b'New document.', 'a.txt')
    assert service.collection.get(include=['documents'])['documents'] == ['Original document.']

def test_storage_failure_rolls_back(setup, monkeypatch):
    _, _, service = setup
    service.ingest(b'Original document.', 'a.txt')
    original = service.collection.upsert
    def fail_after_write(**kwargs):
        original(**kwargs)
        raise RuntimeError('Simulated storage failure')
    monkeypatch.setattr(type(service.collection), 'upsert', lambda self, **kw: fail_after_write(**kw))
    with pytest.raises(RuntimeError):
        service.ingest(b'Replacement document.', 'a.txt')
    assert service.collection.get(include=['documents'])['documents'] == ['Original document.']

def test_filter_delete(setup):
    _, _, service = setup
    service.ingest(b'Gateway MQTT telemetry.', 'gateway.md')
    service.ingest(b'Support hours weekdays.', 'support.md')
    found = service.retrieve('Gateway telemetry', source='support.md')
    assert found and all(s['source'] == 'support.md' for s in found)
    assert service.retrieve('anything', source='missing.md') == []
    assert service.delete('support.md')['deleted_chunks'] == 1
    assert service.delete('support.md')['deleted_chunks'] == 0

def test_citations_and_budget(setup):
    config, provider, service = setup
    config.context_max_chars = 2400
    service.ingest(('gateway telemetry '*300).encode(), 'gateway.md')
    result = service.ask('gateway telemetry', top_k=20)
    assert result['cited_labels'] == ['S1']
    assert result['warnings'] == []
    assert sum(len(s['text']) for s in result['sources']) <= 2400
    provider.answer = 'Unfounded [S99]'
    result = service.ask('gateway')
    assert result['answer'] == UNKNOWN and result['warnings']

def test_persistence_fingerprint(setup):
    config, provider, service = setup
    service.ingest(b'Persist this document.', 'a.txt')
    reopened = RagService(config, provider)
    assert reopened.list_documents() == [{'source':'a.txt', 'chunks':1}]
    config.chunk_size = 1800
    with pytest.raises(ValueError, match='Pipeline configuration changed'):
        RagService(config, provider)

def test_split_overlap_pages():
    chunks = split_pages([(1,'abcdefghij'*100), (2,'second page')], 200, 30)
    assert max(len(c.text) for c in chunks) <= 200
    assert chunks[0].text[-30:] == chunks[1].text[:30]
    assert chunks[-1].page == 2
    assert [c.index for c in chunks] == list(range(len(chunks)))

@pytest.mark.parametrize('text', ['😀'*1000, 'one two\n\n'*400, 'x'*1000])
def test_chunk_progress(text):
    chunks = split_pages([(0,text)], 200, 199)
    assert chunks and all(0 < len(c.text) <= 200 for c in chunks)
    assert len(chunks) <= len(text)

def test_extraction_validation():
    assert extract('Caffè\r\n\r\ntext'.encode(), 'a.md') == [(0,'Caffè\n\ntext')]
    for raw, name in [(b'\xff','a.txt'), (b'  ','a.txt'), (b'bad','a.pdf')]:
        with pytest.raises(ValueError):
            extract(raw,name)
    writer = PdfWriter(); writer.add_blank_page(width=200,height=200)
    raw = BytesIO(); writer.write(raw)
    with pytest.raises(ValueError,match='OCR'):
        extract(raw.getvalue(),'scan.pdf')
    assert source_name('../../a.txt') == 'a.txt'
    assert source_name('C:\\docs\\a.md') == 'a.md'
    with pytest.raises(ValueError):
        source_name('a.exe')
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)

def test_api_auth_validation_upload_chat_static(setup):
    config, _, service = setup
    config.app_api_key = SecretStr('test-token')
    with TestClient(create_app(config,service)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        assert client.get('/api/documents').status_code == 401
        client.headers['Authorization'] = 'Bearer test-token'
        assert client.post('/api/documents', files={'file':('a.txt',b'Gateway buffers 24 hours.')}).status_code == 200
        response = client.post('/api/chat',json={'question':'How long?'})
        assert response.status_code == 200 and response.json()['sources']
        assert client.post('/api/chat',json={'question':'  '}).status_code == 422
        assert client.post('/api/search',json={'question':'q','top_k':21}).status_code == 422
        assert client.post('/api/documents',files={'file':('a.exe',b'bad')}).status_code == 400
        assert client.request('DELETE','/api/documents',json={'source':'a.txt'}).json()['deleted_chunks'] == 1
        config.max_upload_bytes = 10
        assert client.post('/api/documents',files={'file':('a.txt',b'x'*11)}).status_code == 413

def test_metrics():
    assert retrieval_metrics([{'source':'a'},{'source':'a'},{'source':'b'}], ['b','c']) == {
        'hit_at_k':1.,'source_recall_at_k':0.5,'reciprocal_rank':0.5}
    assert retrieval_metrics([],[]) is None

def test_openai_sdk_contract():
    seen = []
    def handler(request):
        body = json.loads(request.content); seen.append(body)
        if request.url.path.endswith('/embeddings'):
            return httpx.Response(200,json={'object':'list','model':'text-embedding-3-small',
                'data':[{'object':'embedding','index':0,'embedding':[1.,0.]}],
                'usage':{'prompt_tokens':2,'total_tokens':2}})
        return httpx.Response(200,json={'id':'resp_test','object':'response','created_at':0,
            'status':'completed','model':'gpt-4.1-mini','error':None,'incomplete_details':None,
            'output':[{'id':'msg_test','type':'message','role':'assistant','status':'completed',
            'content':[{'type':'output_text','text':'Answer [S1]','annotations':[]}]}]})
    provider = OpenAIProvider(Settings(_env_file=None,openai_api_key='test-key'))
    provider._client = OpenAI(api_key='test-key',http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.embed(['test']) == [[1.,0.]]
    assert provider.generate('Question',[{'label':'S1','source':'a','page':0,'text':'evidence'}]) == 'Answer [S1]'
    assert seen[-1]['store'] is False and 'untrusted' in seen[-1]['instructions']
    provider._client.close()
