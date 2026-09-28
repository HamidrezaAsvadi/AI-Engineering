"""RAG orchestration. Chroma is embedded in ONE API process."""
from hashlib import sha256
import re
from threading import RLock
from time import perf_counter
import chromadb
from .config import Settings
from .documents import digest, extract, source_name, split_pages

UNKNOWN = "I don't know based on the provided documents."

class RagService:
    def __init__(self, settings: Settings, provider):
        self.settings, self.provider = settings, provider
        self.lock = RLock()
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        # Reject silent reuse after an embedding/chunking pipeline change.
        signature = f'v1|{settings.embedding_model}|{settings.chunk_size}|{settings.chunk_overlap}|cosine'
        self.fingerprint = sha256(signature.encode()).hexdigest()
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=None,
            configuration={'hnsw': {'space': 'cosine'}},
            metadata={'pipeline_fingerprint': self.fingerprint},
        )
        if (self.collection.metadata or {}).get('pipeline_fingerprint') != self.fingerprint:
            raise ValueError('Pipeline configuration changed. Use a new COLLECTION_NAME and re-ingest documents')

    def ingest(self, raw: bytes, filename: str) -> dict:
        source = source_name(filename)
        if len(raw) > self.settings.max_upload_bytes:
            raise ValueError('Maximum file size is 10 MiB')
        version = digest(raw)
        with self.lock:
            old = self.collection.get(where={'source': source}, include=['metadatas'])
            if old['ids'] and all(m['file_hash'] == version for m in old['metadatas']):
                return {'source': source, 'chunks': len(old['ids']), 'status': 'unchanged'}
            chunks = split_pages(extract(raw, source), self.settings.chunk_size, self.settings.chunk_overlap)
            texts = [c.text for c in chunks]
            # Finish external calls BEFORE touching the existing stored document.
            vectors = self.provider.embed(texts)
            if len(vectors) != len(chunks):
                raise RuntimeError('Embedding provider returned the wrong number of vectors')
            ids = [sha256(f'{source}|{version}|{c.index}'.encode()).hexdigest() for c in chunks]
            metadata = [{'source': source, 'page': c.page, 'chunk_index': c.index, 'file_hash': version}
                        for c in chunks]
            old_ids = set(old['ids'])
            new_ids = set(ids) - old_ids
            batch_size = min(128, self.client.get_max_batch_size())
            try:
                for start in range(0, len(ids), batch_size):
                    end = start + batch_size
                    self.collection.upsert(ids=ids[start:end], documents=texts[start:end],
                                           embeddings=vectors[start:end], metadatas=metadata[start:end])
            except Exception:
                # Best effort rollback of this version; the old version remains.
                if new_ids:
                    self.collection.delete(ids=list(new_ids))
                raise
            stale = old_ids - set(ids)
            if stale:
                self.collection.delete(ids=list(stale))
            return {'source': source, 'chunks': len(ids), 'status': 'indexed'}

    def list_documents(self) -> list[dict]:
        with self.lock:
            grouped = {}
            offset = 0
            while True:
                batch = self.collection.get(limit=500, offset=offset, include=['metadatas'])
                for m in batch['metadatas']:
                    grouped[m['source']] = grouped.get(m['source'], 0) + 1
                if len(batch['ids']) < 500:
                    break
                offset += 500
        return [{'source': source, 'chunks': count} for source, count in sorted(grouped.items())]

    def delete(self, source: str) -> dict:
        with self.lock:
            existing = self.collection.get(where={'source': source}, include=[])
            if existing['ids']:
                self.collection.delete(ids=existing['ids'])
        return {'source': source, 'deleted_chunks': len(existing['ids'])}

    def retrieve(self, question: str, top_k: int | None = None, source: str | None = None) -> list[dict]:
        k = top_k if top_k is not None else self.settings.top_k
        with self.lock:
            count = self.collection.count()
            if count == 0:
                return []
            vector = self.provider.embed([question])[0]
            args = {'query_embeddings': [vector], 'n_results': min(k, count),
                    'include': ['documents', 'metadatas', 'distances']}
            if source:
                args['where'] = {'source': source}
            result = self.collection.query(**args)
        sources = []
        for id_, text, meta, distance in zip(result['ids'][0], result['documents'][0],
                                              result['metadatas'][0], result['distances'][0]):
            if self.settings.max_distance is not None and distance > self.settings.max_distance:
                continue
            sources.append({'id': id_, 'label': f'S{len(sources) + 1}', 'text': text,
                            'source': meta['source'], 'page': meta['page'],
                            'chunk_index': meta['chunk_index'], 'distance': float(distance)})
        return sources

    def ask(self, question: str, top_k: int | None = None, source: str | None = None) -> dict:
        start = perf_counter()
        candidates = self.retrieve(question, top_k, source)
        # Include whole chunks only; every returned passage was passed to the LLM.
        sources, used = [], 0
        for candidate in candidates:
            if used + len(candidate['text']) > self.settings.context_max_chars:
                break
            sources.append(candidate)
            used += len(candidate['text'])
        answer = self.provider.generate(question, sources) if sources else UNKNOWN
        labels = set(re.findall(r'\[(S\d+)\]', answer))
        allowed = {s['label'] for s in sources}
        warnings = []
        if labels - allowed:
            warnings.append('Model returned an invalid citation; answer was withheld.')
            answer = UNKNOWN
            labels = set()
        elif sources and answer != UNKNOWN and not labels:
            warnings.append('No source citations in the answer; verify it against the passages.')
        return {'answer': answer, 'sources': sources, 'cited_labels': sorted(labels),
                'warnings': warnings, 'latency_ms': round((perf_counter() - start) * 1000, 1)}
