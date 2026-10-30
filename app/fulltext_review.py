"""Load the publication-based review, retaining excluded records for the audit view."""
import json
from pathlib import Path


def apply_fulltext_review(catalog):
    review = json.loads((Path(__file__).parent / 'data/fulltext_review.json').read_text(encoding='utf-8'))
    current_ids = {p['id'] for p in catalog['papers']}
    reviewed_ids = {p['id'] for p in review['papers']}
    if current_ids != reviewed_ids:
        raise ValueError('正式清单的论文ID已变动，请重新核查后更新应用数据。不能把旧核查结果套到新清单。')
    for key in ('papers', 'models', 'pairs', 'dataset_records', 'review_meta', 'schema_version', 'reviewed_papers', 'archived_papers'):
        catalog[key] = review[key]
    catalog['enrichment'] = dict(updated_at=review['review_meta']['updated_at'], scope=review['review_meta']['scope'])
    catalog['audit'] = [p for p in catalog['papers'] if p['group'] == '应用类']
    return catalog
