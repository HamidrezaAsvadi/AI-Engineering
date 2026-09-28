"""Small labelled smoke evaluation, not a production benchmark."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
import httpx
from app.config import Settings
from app.rag import UNKNOWN


def retrieval_metrics(sources, relevant):
    # Document-level ranking: repeated chunks from the same file count once.
    ranked = list(dict.fromkeys(s['source'] for s in sources))
    relevant = set(relevant)
    if not relevant:
        return None
    found = relevant.intersection(ranked)
    ranks = [i + 1 for i, source in enumerate(ranked) if source in relevant]
    return {'hit_at_k': float(bool(found)), 'source_recall_at_k': len(found) / len(relevant),
            'reciprocal_rank': 1 / min(ranks) if ranks else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('evaluation/questions.jsonl'))
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--generate', action='store_true', help='Also call LLM; extra API cost')
    parser.add_argument('--output', type=Path, default=Path('evaluation/results.json'))
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error('--top-k must be between 1 and 20')
    settings = Settings()
    token = settings.app_api_key.get_secret_value()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    rows = []
    with httpx.Client(base_url=settings.rag_api_url, headers=headers, timeout=300, trust_env=False) as client:
        for record in records:
            start = perf_counter()
            response = client.post('/api/chat' if args.generate else '/api/search',
                                   json={'question': record['question'], 'top_k': args.top_k})
            response.raise_for_status()
            result = response.json()
            row = {'question': record['question'], 'latency_ms': round((perf_counter()-start)*1000, 1),
                   'retrieved_sources': [s['source'] for s in result['sources']],
                   'metrics': retrieval_metrics(result['sources'], record['relevant_sources'])}
            if args.generate:
                answer = result['answer']
                row['answer'] = answer
                row['warnings'] = result['warnings']
                row['abstained'] = answer.strip() == UNKNOWN
                row['expected_abstention'] = not record['answerable']
                terms = record['expected_terms']
                row['expected_term_coverage'] = (sum(t.lower() in answer.lower() for t in terms) / len(terms)) if terms else None
            rows.append(row)
    metrics = [r['metrics'] for r in rows if r['metrics'] is not None]
    summary = {key: mean(m[key] for m in metrics) for key in ('hit_at_k', 'source_recall_at_k', 'reciprocal_rank')} if metrics else {}
    summary['mean_latency_ms'] = mean(r['latency_ms'] for r in rows) if rows else 0
    if args.generate and rows:
        summary['abstention_decision_accuracy'] = mean(r['abstained'] == r['expected_abstention'] for r in rows)
    report = {'top_k': args.top_k, 'generation_enabled': args.generate, 'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f'Detailed report: {args.output}')

if __name__ == '__main__':
    main()
