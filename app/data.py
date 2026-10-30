"""Read-only evidence model. No upstream files are changed or executed."""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import openpyxl
from enrichment import apply_enrichment
from fulltext_review import apply_fulltext_review

APP_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/Users/paris404/Desktop/streetscape_review')
FILES = {
    'papers': 'analysis/two_waves/papers_two_waves.csv',
    'abstracts': 'app_group/app_group_abstract_extraction.csv',
    'fulltext': 'app_group/app_group_fulltext_extraction.csv',
    'models': 'app_group/app_group_models_long.csv',
    'screening': 'screening_ft/fulltext_final.csv',
    'datasets': 'datasets/数据集表_v2.xlsx',
}
TYPE_NAMES = {'application': '应用类', 'algorithm': '算法开发', 'transfer': '迁移与适配',
              'comparison': '模型比较', 'dataset': '数据集研究'}
TASK_NAMES = {'semantic segmentation': '语义分割', 'object detection': '目标检测',
              'instance segmentation': '实例分割', 'panoptic segmentation': '全景分割',
              'scene-text detection': '场景文字检测', 'change detection': '变化检测',
              'other': '其他任务', 'unspecified': '任务未说明', '': '任务未说明'}
THEMES = {
    'perception & aesthetics': '感知与美学', 'health & wellbeing': '健康与福祉',
    'walkability & active travel': '步行性与主动出行', 'transport & road safety': '交通与道路安全',
    'greenery & vegetation': '绿化与植被', 'thermal environment & climate': '热环境与气候',
    'housing & economy': '住房与经济', 'crime, disorder & social': '犯罪、失序与社会',
    'equity & environmental justice': '公平与环境正义', 'land use & urban function': '土地利用与城市功能',
    'heritage & tourism': '遗产与旅游', 'urban form & morphology': '城市形态',
    'environmental quality (air, noise, light)': '环境质量', 'infrastructure & assets': '基础设施与资产',
    'other': '其他主题', '': '主题未编码',
}
DATASET_PATTERNS = [
    ('Cityscapes', r'cityscape'), ('ADE20K', r'ade\s?20k'), ('Mapillary Vistas', r'mapillary'),
    ('COCO', r'\bcoco'), ('CamVid', r'camvid'), ('KITTI', r'\bkitti'), ('BDD100K', r'\bbdd'),
    ('PASCAL VOC', r'pascal|\bvoc\b'), ('SA-1B', r'sa-1b'), ('Objects365', r'objects365|obj365'),
    ('ApolloScape', r'apollo'), ('IDD', r'\bidd\b|india driving'), ('ImageNet', r'imagenet'),
    ('Places', r'places'), ('Open Images', r'open\s?images'),
]


def split(text):
    return sorted(set(x.strip() for x in (text or '').split(';') if x.strip()))


def dataset_name(text):
    for name, pat in DATASET_PATTERNS:
        if re.search(pat, text or '', re.I):
            return name
    return (text or '').strip()


def datasets(text):
    return sorted(set(dataset_name(x) for x in split(text)
                      if x.lower() not in ('none', 'not stated', 'unknown', 'unspecified')))


def tasks(text):
    vals = [TASK_NAMES.get(x, x) for x in split(text)]
    return sorted(set(vals)) or ['任务未说明']


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def get_source():
    return Path(os.environ.get('STREETSCAPE_SOURCE', str(DEFAULT_SOURCE))).expanduser()


def fingerprint():
    paths = [APP_DIR / 'data/review.json', APP_DIR / 'data/fulltext_review.json']
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def build_catalog(root):
    source = {k: csv_rows(root / f) for k, f in FILES.items() if f.endswith('.csv')}
    final = [p for p in source['papers'] if p['formal_included'].startswith('yes')]
    counts = Counter(p['rec_id'] for p in final)
    if any(v > 1 for v in counts.values()):
        raise ValueError('正式论文清单存在重复 ID，需先修正来源。')
    bad = {p['type'] for p in final} - TYPE_NAMES.keys()
    if bad:
        raise ValueError(f'出现未映射的研究类型：{bad}')
    abstracts = {r['rec_id']: r for r in source['abstracts']}
    fulltext = {r['rec_id']: r for r in source['fulltext']}
    screened = {r['rec_id']: r for r in source['screening']}
    mains = defaultdict(list)
    for n, row in enumerate(source['models'], 2):
        if row['role'] == 'main':
            mains[row['rec_id']].append(dict(row, source_row=n))
    papers, model_rows, paired = [], [], []
    for source_row, p in enumerate(source['papers'], 2):
        if not p['formal_included'].startswith('yes'):
            continue
        rid = p['rec_id']; a = abstracts.get(rid, {}); f = fulltext.get(rid, {}); s = screened.get(rid, {})
        use_main = p['source'] == 'app_group_fulltext'
        actual = mains.get(rid, []) if use_main else []
        paper_tasks = sorted({t for m in actual for t in tasks(m['task'])}) if actual else tasks(s.get('task') or a.get('task'))
        if len(paper_tasks) > 1 and '任务未说明' in paper_tasks:
            paper_tasks.remove('任务未说明')
        model_names = sorted({m['model'] for m in actual if m['model']}) if actual else split(p['main_models'])
        paper_datasets = datasets(p['pretrained_on'])
        theme = p['theme'] or a.get('theme', '')
        doi = f.get('doi') or s.get('doi') or a.get('doi') or ''
        evidence = f.get('evidence') if use_main else (s.get('f1_evidence') or a.get('evidence', ''))
        paper = dict(id=rid, year=int(p['year']) if p['year'].isdigit() else None, title=p['title'],
                     group='应用类' if p['type'] == 'application' else '算法与方法类',
                     raw_type=p['type'], type_name=TYPE_NAMES[p['type']], wave=p['wave'], tasks=paper_tasks,
                     theme=THEMES.get(theme, theme or '主题未编码'), models=model_names, datasets=paper_datasets,
                     imagery=p['imagery_source'], evidence=evidence or '', doi=doi,
                     level='仅摘要' if p['source'] == 'app_group_abstract_only' else '全文',
                     source=p['source'], source_row=source_row, location=f.get('location') or a.get('location', ''),
                     classes=f.get('classes_used', ''), indicators=f.get('indicators') or a.get('indicators', ''),
                     framework=f.get('framework') or s.get('framework', ''), classification_note=p['wave_note'])
        papers.append(paper)
        if actual:
            for m in actual:
                mr = dict(id=rid, model=m['model'], raw=m['model_raw'], tasks=tasks(m['task']),
                          datasets=datasets(m['pretrained_on']), scope='模型级明确对应',
                          source=FILES['models'], source_row=m['source_row'])
                model_rows.append(mr)
                paired.extend(dict(id=rid, model=mr['model'], dataset=d, scope=mr['scope']) for d in mr['datasets'])
        else:
            for name in model_names:
                single = len(model_names) == 1
                scope = '单模型论文对应' if single else '论文级记录，未建立对应'
                mr = dict(id=rid, model=name, raw=name, tasks=paper_tasks if single else [],
                          datasets=paper_datasets if single else [], scope=scope, source=FILES['papers'], source_row=source_row)
                model_rows.append(mr)
                if single:
                    paired.extend(dict(id=rid, model=name, dataset=d, scope=scope) for d in paper_datasets)
    wb = openpyxl.load_workbook(root / FILES['datasets'], read_only=True, data_only=True)
    tables = {ws.title: list(ws.values) for ws in wb}
    catalogue = []
    headers = tables['1 数据集总表'][0]
    for row_no, row in enumerate(tables['1 数据集总表'][1:], 2):
        r = dict(zip(headers, row)); name = dataset_name(r['数据集']); annotation = r['标注类型'] or ''
        dtasks = []
        for word, label in [('检测', '目标检测'), ('框', '目标检测'), ('语义', '语义分割'), ('实例', '实例分割'), ('全景', '全景分割')]:
            if word in annotation and label not in dtasks:
                dtasks.append(label)
        if name == 'SA-1B':
            dtasks = ['无类别掩码分割']
        catalogue.append(dict(name=name, full_name=r['数据集'], year=r['年份'], reference=r['出处'],
                              imagery=r['影像类型'], scene=r['场景（室内/室外）'], size=r['规模与划分'],
                              resolution=r['分辨率'], classes=r['类别数'], annotation=annotation,
                              license=r['许可'], fit=r['与街景研究的贴合度'], geography=r['采集地区/背景'],
                              tasks=dtasks, source_row=row_no))
    classlists = [dict(dataset=dataset_name(r[0]), variant=r[0], subset=r[1], count=r[2], labels=r[3], source_row=i)
                  for i, r in enumerate(tables['2 类别清单'][1:], 2)]
    coverage = []
    chead = tables['7 城市要素覆盖矩阵'][0]
    for row_no, row in enumerate(tables['7 城市要素覆盖矩阵'][1:], 2):
        if not row[1]:
            continue
        for col, variant in enumerate(chead[2:], 2):
            label = row[col] or ''
            code = 2 if label.startswith('✓') else 1 if label.startswith('≈') else 0
            coverage.append(dict(element=row[0], english=row[1], dataset=dataset_name(variant),
                                 variant=variant, coverage=code, labels=label,
                                 source_cell=f'{openpyxl.utils.get_column_letter(col+1)}{row_no}'))
    sources = [dict(file=f, updated=datetime.fromtimestamp((root / f).stat().st_mtime, ZoneInfo('Australia/Brisbane')).isoformat())
               for f in FILES.values()]
    return apply_fulltext_review(apply_enrichment(dict(papers=papers, models=model_rows, pairs=paired, datasets=catalogue, classlists=classlists,
                coverage=coverage, sources=sources, mode='live', schema_version=1)))


def load_catalog():
    # A reviewed corpus must not drift as an upstream extraction is being edited.
    data = json.loads((APP_DIR / 'data' / 'review.json').read_text(encoding='utf-8'))
    data['mode'] = 'reviewed_snapshot'
    return data


if __name__ == '__main__':
    target = APP_DIR / 'data' / 'review.json'
    target.parent.mkdir(exist_ok=True)
    content = build_catalog(get_source())
    target.write_text(json.dumps(content, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'snapshot': str(target), 'types': dict(Counter(p['group'] for p in content['papers']))}, ensure_ascii=False))
