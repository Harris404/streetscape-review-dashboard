"""Shared model-name standardisation for app.py and report_build.py (2026-10-30).

- Family names: one vocabulary for both the evidence snapshot (catalog) and the wave-2 extraction; YOLO is split by version.
- 算法与方法类: the paper's base model comes from the wave-2 full-text extraction (same as the 训练与迁移 page);
  other models the authors ran become "comparison".
- Variant names: "family (backbone / size)" only; spelling, case and training settings (crop size etc.) are dropped.
"""
from __future__ import annotations

import re

import pandas as pd

QUALIFIED = '模型／模型家族'
ALIASES = {
    'hrnet/ocr': 'HRNet(+OCR)', 'hrnet(+ocr)': 'HRNet(+OCR)', 'hrnet': 'HRNet(+OCR)', 'ocrnet': 'HRNet(+OCR)',
    'sam family': 'SAM', 'sam': 'SAM', 'deeplabv3+': 'DeepLabv3+', 'deeplabv3': 'DeepLabv3', 'deeplab (unspecified)': 'DeepLab（版本未注明）',
    'deeplab v1/v2': 'DeepLab v1/v2', 'detr family': 'DETR 系列', 'swin (other head)': 'Swin（其他检测/分割头）',
    'vit (other head)': 'ViT（其他头）', '通用 cnn / 自建网络': '自建／通用 CNN', 'classical ml / 非深度': '传统机器学习（非深度）',
    'yolo (version n/s)': 'YOLO（版本未注明）', 'yolo': 'YOLO（版本未注明）', 'panoptic-deeplab/fpn': 'Panoptic FPN', 'panoptic-fpn': 'Panoptic FPN',
    'mask r-cnn': 'Mask R-CNN', 'faster r-cnn': 'Faster R-CNN', 'upernet': 'UperNet',
}
NOT_NAMED = {'其他 / 未说明', 'unnamed', ''}
BACKBONE = [
    (r'resnext-?\s?(\d+)', 'ResNeXt-{}'), (r'(?:resnet|\br)-?\s?(\d{2,3})(?:-d8|-fpn|\b)', 'ResNet-{}'), (r'xception-?\s?(\d+)', 'Xception-{}'),
    (r'mobilenet\s?-?v?(\d)', 'MobileNetV{}'), (r'hrnet-?w?(\d{2})', 'HRNet-W{}'), (r'swin-?\s?([tsbl])\b', 'Swin-{}'),
    (r'mit-?\s?(b\d)', 'MiT-{}'), (r'segformer-?\s?(b\d)', 'MiT-{}'), (r'vit-?\s?([bslh])(?:/\d+)?\b', 'ViT-{}'), (r'vgg-?\s?(\d{2})', 'VGG-{}'),
    (r'efficientnet-?\s?(b\d)', 'EfficientNet-{}'), (r'darknet-?\s?(\d+)', 'DarkNet-{}'), (r'convnext-?\s?(\w+)', 'ConvNeXt-{}'),
    (r'wide\s?resnet-?\s?(\d+)', 'WideResNet-{}'),
]


def unify(name: str, raw: str = '') -> str:
    """Common family name; YOLO split by version using the name or the raw description."""
    n = str(name or '').strip()
    text = f'{n} {raw or ""}'.lower()
    if 'pp-yolo' in text:
        return 'PP-YOLOE'
    if 'yolo-world' in text:
        return 'YOLO-World'
    if 'yolo' in text and 'yolox' not in text and 'yolo-nas' not in text:
        m = re.search(r'yolo\s*-?v?\s*(\d{1,2})', text)
        if m:
            v = int(m.group(1))
            return f'YOLO{v}' if v >= 11 else f'YOLOv{v}'
        return 'YOLO（版本未注明）'
    if 'resnet50-dilated + ppm' in text or 'ppm_deepsup' in text or 'ppm-deepsup' in text:
        return 'PSPNet'
    return ALIASES.get(n.lower(), n)


def std_variant(raw: str, family: str) -> str:
    """'family (backbone)' with YOLO size letter kept; drops spelling/case/training-setting differences."""
    t = str(raw or '').lower().replace('_', '-')
    if family.startswith('YOLO') and '未注明' not in family:
        m = re.search(r'yolo\s*-?v?\s*\d{1,2}\s*-?([nsmlx]\d?)\b', t)
        return family + (m.group(1).lower() if m else '')
    for rx, fmt in BACKBONE:
        m = re.search(rx, t)
        if m:
            g = m.group(1)
            return f'{family} ({fmt.format(g.upper() if fmt.endswith("-{}") and g.isalpha() else g)})'
    return family


def stem(family: str) -> str:
    f = str(family)
    return 'YOLO' if f.startswith('YOLO') else 'DeepLab' if f.startswith('DeepLab') else f


def kind_of(family: str, model_kind) -> str:
    if family in NOT_NAMED:
        return '未具名'
    if family == '传统机器学习（非深度）':
        return '传统视觉方法'
    if family == '自建／通用 CNN':
        return QUALIFIED
    return model_kind(family)


def standardise(models: pd.DataFrame, pairs: pd.DataFrame, records: pd.DataFrame, papers: pd.DataFrame,
                flat: pd.DataFrame, comp_rows: list[dict], norm, model_kind, enet_ok: set | None = None,
                raw_models: dict | None = None, official_models: dict | None = None):
    """Return (models, pairs, records) with unified family names, standardised variants and split roles.
    enet_ok: papers that really use ENet. The snapshot's old name rule mapped any name containing 'enet'
    (MobileNet, BiSeNet, SpineNet, RefineNet ...) to ENet; those rows are dropped."""
    models = models.copy(); pairs = pairs.copy(); records = records.copy()
    if enet_ok is not None:
        bad = (models.model == 'ENet') & ~models.id.isin(enet_ok)
        models = models[~bad]
        pairs = pairs[~((pairs.model == 'ENet') & ~pairs.id.isin(enet_ok))] if len(pairs) else pairs
        records = records[~((records.model == 'ENet') & ~records.id.isin(enet_ok))] if len(records) else records
    meth = set(papers[papers.group == '算法与方法类'].id); app = set(papers[papers.group == '应用类'].id)
    base = {}
    for r in flat.itertuples():
        fam = getattr(r, 'family', '') or ''
        fam = unify(norm(r.base_model), r.base_model) if fam in NOT_NAMED or not isinstance(fam, str) else unify(fam, r.base_model)
        base[r.id] = (fam, r.base_model if isinstance(r.base_model, str) else '')
    rename = {}; aux = []
    for i, row in models.iterrows():
        raw = ' '.join(str(row.get(k, '') or '') for k in ('variant', 'raw'))
        fam = unify(row['model'], raw)
        if fam == 'YOLO（版本未注明）' and raw_models:
            # recover the version from the paper's raw model names; YOLO used only as an image classifier is auxiliary
            ys = [r for r in raw_models.get(row['id'], []) if 'yolo' in r.lower()]
            det = [r for r in ys if 'classif' not in r.lower()]
            if ys and not det:
                aux.append(i)
            elif det:
                fam = unify(det[0], det[0]); models.at[i, 'raw'] = det[0]
        rename[i] = fam
    models['family_new'] = pd.Series(rename)
    models.loc[aux, 'role'] = 'auxiliary'
    # method papers: base model from the wave-2 extraction is "main", everything else the authors ran is "comparison"
    new_rows = []
    for pid in meth:
        if pid not in base:
            continue
        fam, braw = base[pid]
        idx = models.index[(models.id == pid) & models.role.isin(['main', 'comparison'])]
        models.loc[idx, 'role'] = 'comparison'
        # a catalog row is the base model if it has the same family, or is the same family without a version (e.g. 'YOLO' for YOLOv5)
        hit = [i for i in idx if models.at[i, 'family_new'] == fam
               or ('未注明' in models.at[i, 'family_new'] and stem(models.at[i, 'family_new']) == stem(fam))]
        if hit:
            models.loc[hit, 'role'] = 'main'
            models.loc[hit, 'family_new'] = fam
            models.loc[hit, 'variant'] = braw or models.loc[hit, 'variant']
        else:
            new_rows.append(dict(id=pid, model=fam, family_new=fam, raw=braw, variant=braw, tasks=[], datasets=[], scope='第二波专项提取',
                                 source='wave2_flat.csv', source_row=0, kind=kind_of(fam, model_kind), evidence_status='第二波全文提取',
                                 role='main', dataset_roles=[]))
    if new_rows:
        models = pd.concat([models, pd.DataFrame(new_rows)], ignore_index=True)
    # papers without any named main model: take the official list's main model names (recovered by full-text checks)
    if official_models:
        named = set(models.loc[(models.role == 'main') & (models.kind.fillna(QUALIFIED) == QUALIFIED), 'id'])
        add = []
        for pid in app - named:
            for nm in official_models.get(pid, []):
                fam = unify(nm, nm)
                if fam and fam not in NOT_NAMED:
                    add.append(dict(id=pid, model=fam, family_new=fam, raw=nm, variant=nm, tasks=[], datasets=[], scope='正式清单补录',
                                    source='papers_two_waves.csv', source_row=0, kind=kind_of(fam, model_kind), evidence_status='全文核查补录',
                                    role='main', dataset_roles=[]))
        if add:
            models = pd.concat([models, pd.DataFrame(add)], ignore_index=True)
    # application papers: add comparison models from the app-group full-text extraction
    have = set(zip(models.id, models.family_new)); extra = []
    for r in comp_rows:
        fam = unify(r['model'], r.get('model_raw', ''))
        if r['rec_id'] in app and (r['rec_id'], fam) not in have and r['model'] != 'unnamed':
            have.add((r['rec_id'], fam))
            task = {'semantic segmentation': '语义分割', 'object detection': '目标检测', 'instance segmentation': '实例分割',
                    'panoptic segmentation': '全景分割'}.get(r.get('task', ''), '其他任务')
            extra.append(dict(id=r['rec_id'], model=fam, family_new=fam, raw=r.get('model_raw', ''), tasks=[task], datasets=[],
                              scope='模型级明确对应', source='app_group_models_long.csv', source_row=0, kind=kind_of(fam, model_kind),
                              evidence_status='应用组全文提取', variant=r.get('model_raw', ''), role='comparison', dataset_roles=[]))
    if extra:
        models = pd.concat([models, pd.DataFrame(extra)], ignore_index=True)
    # keep old->new names for pairs / dataset records, then finalise columns
    old2new = {(r.id, r.model): r.family_new for r in models.itertuples()}
    roles = {(r.id, r.family_new): r.role for r in models.itertuples()}
    models['model'] = models['family_new']
    models['kind'] = [k if k != QUALIFIED else kind_of(f, model_kind) for k, f in zip(models.kind.fillna(QUALIFIED), models.model)]
    models['variant'] = [std_variant(f'{v} {r}', f) for v, r, f in zip(models.variant.fillna(''), models.raw.fillna(''), models.model)]
    models = models.drop(columns=['family_new']).drop_duplicates(['id', 'model', 'variant', 'role'])
    for df in (pairs, records):
        if len(df):
            df['model'] = [old2new.get((i, m), unify(m)) if m else m for i, m in zip(df.id, df.model)]
            df['model_role'] = [roles.get((i, m), r) if m and r != 'paper' else r for i, m, r in zip(df.id, df.model, df.model_role)]
    return models, pairs, records


DATASET_ALIAS = {'COCO Panoptic': 'COCO', 'COCO-Stuff': 'COCO', 'OpenImages': 'Open Images'}
# research themes for the 19 untrained algorithm / comparison papers now counted as 应用类 (coded from title and full-text task, 2026-10-30)
THEME_FIX = {
    'R0311': '城市形态', 'R0665': '热环境与气候', 'R1852': '基础设施与资产', 'R2423': '住房与经济', 'R0139': '步行性与主动出行',
    'R0690': '基础设施与资产', 'R2161': '绿化与植被', 'R2373': '绿化与植被', 'R3010': '城市形态', 'R3285': '城市形态', 'R1439': '城市形态',
    'R0108': '现成模型评测与工具', 'R0480': '现成模型评测与工具', 'R2028': '现成模型评测与工具', 'R2744': '现成模型评测与工具',
    'R1318': '现成模型评测与工具', 'R1336': '现成模型评测与工具', 'R1601': '现成模型评测与工具', 'R2083': '现成模型评测与工具',
}


def fix_paper_fields(papers: pd.DataFrame) -> pd.DataFrame:
    papers = papers.copy()
    papers['datasets'] = papers.datasets.apply(lambda ds: sorted({DATASET_ALIAS.get(d, d) for d in ds}))
    papers['theme'] = [THEME_FIX.get(i, t) if g == '应用类' else t for i, t, g in zip(papers.id, papers.theme, papers.group)]
    return papers


def fix_dataset_names(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if len(df) and col in df:
        df = df.copy(); df[col] = df[col].map(lambda d: DATASET_ALIAS.get(d, d))
    return df
