"""All commands use HTTP: only the API process opens Chroma."""
import argparse
import json
from pathlib import Path
import httpx
from .config import Settings


def main():
    parser = argparse.ArgumentParser(description='RAG client; start the API first')
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest')
    ingest.add_argument('path', type=Path)
    ask = sub.add_parser('ask')
    ask.add_argument('question')
    ask.add_argument('--source')
    ask.add_argument('--top-k', type=int, default=5)
    ask.add_argument('--search-only', action='store_true')
    sub.add_parser('list')
    delete = sub.add_parser('delete')
    delete.add_argument('source')
    args = parser.parse_args()
    settings = Settings()
    token = settings.app_api_key.get_secret_value()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    try:
        with httpx.Client(base_url=settings.rag_api_url, headers=headers, timeout=300, trust_env=False) as client:
            def show(response):
                response.raise_for_status()
                print(json.dumps(response.json(), ensure_ascii=False, indent=2))
            if args.command == 'ingest':
                files = [args.path] if args.path.is_file() else sorted(
                    p for p in args.path.rglob('*') if p.is_file() and p.suffix.lower() in {'.pdf', '.txt', '.md'})
                if not files:
                    parser.error('No PDF, TXT, or MD files found')
                names = [p.name for p in files]
                if len(names) != len(set(names)):
                    parser.error('Duplicate basenames in this directory; rename files before ingestion')
                for path in files:
                    with path.open('rb') as stream:
                        show(client.post('/api/documents', files={'file': (path.name, stream)}))
            elif args.command == 'ask':
                route = '/api/search' if args.search_only else '/api/chat'
                show(client.post(route, json={'question': args.question, 'source': args.source, 'top_k': args.top_k}))
            elif args.command == 'list':
                show(client.get('/api/documents'))
            else:
                show(client.request('DELETE', '/api/documents', json={'source': args.source}))
    except httpx.HTTPStatusError as exc:
        raise SystemExit(f'API error {exc.response.status_code}: {exc.response.text}') from None
    except httpx.RequestError as exc:
        raise SystemExit(f'Cannot reach API: {exc}. Start uvicorn first.') from None

if __name__ == '__main__':
    main()
