"""Evidence-backed additions; preserve the original extraction for comparison."""
import json
from pathlib import Path

QUALIFIED = '模型／模型家族'
TOOLS = {
    'Detectron2 (unspecified)', 'unspecified MMSegmentation model (ADE20K-trained)',
    'GluonCV ADE20K-pretrained semantic segmentation model (architecture not stated)',
    'pre-trained semantic segmentation network (unspecified, MATLAB)',
    'Mapillary traffic-sign detection (semantic segmentation + patch classifier)',
    'MMSegmentation ensemble of 32 Cityscapes-trained models (individual models listed only in Appendix 1)',
    'ZenSVI segmentation (CNN)', 'Google Cloud Vision API (OCR text detection)',
    'NUS Global Streetscapes (precomputed segmentation)', 'Treepedia', 'GluonCV',
    'CUG.HPSCIL laboratory image semantic recognition software', 'Mapillary API semantic segmentation',
    'PaddleOCRv4', 'PaddleOCR v2.9 (text detection + recognition)', 'EasyOCR',
}
PARTIAL = {
    'ViT-based facade segmentation/material classification pipeline (prior work)',
    'Swin (head unspecified)', 'ResNet18-based bounding-box retrieval (pretrained)',
    'Xception', 'pedestrian detector from Liu et al. (2023) (unspecified)',
}
CLASSICAL = {'Sobel filter + flood-fill sky detection',
             'Hoiem et al. geometric context segmentation (superpixels)', 'GrabCut'}


def named_record(name):
    return bool(name and name.lower() not in {'unnamed', 'none', 'unknown', 'not stated', 'unspecified'})


def model_kind(name):
    if not named_record(name):
        return '未具名'
    if name in TOOLS:
        return '仅工具／服务／预计算结果'
    if name in PARTIAL:
        return '骨干／方法描述，架构未完整说明'
    if name in CLASSICAL:
        return '传统视觉方法'
    return QUALIFIED


def apply_enrichment(catalog):
    path = Path(__file__).parent / 'data' / 'enrichment.json'
    if not path.exists():
        raise FileNotFoundError('缺少 data/enrichment.json，请使用完整应用包。')
    supplement = json.loads(path.read_text(encoding='utf-8'))
    updates = {r['id']: r for r in supplement['updates']}
    audits = {r['id']: r for r in supplement['audit']}
    models_by_id = {}
    for row in catalog['models']:
        models_by_id.setdefault(row['id'], []).append(row)
    final_models, audit_rows = [], []
    for p in catalog['papers']:
        rid = p['id']
        rows = models_by_id.get(rid, [])
        p['original_models'] = list(p['models'])
        p['original_tasks'] = list(p['tasks'])
        p['original_theme'] = p['theme']
        p['original_level'] = p['level']
        p['theme_origin'] = '原有编码'
        p['theme_detail'] = ''
        p['comparison_models'] = []
        p['excluded_model_records'] = []
        p['model_evidence_status'] = '原有提取'
        p['enrichment_notes'] = []
        p['enrichment_sources'] = []
        u = updates.get(rid)
        if u:
            changes = u['changes']
            p['enrichment_notes'] = u['notes']
            p['enrichment_sources'] = u['sources']
            # Never silently overwrite an upstream re-extraction with an older correction.
            baseline = audits[rid]
            conflict = any(p[key] != baseline['original_' + key]
                           for key in ('models', 'tasks', 'theme'))
            if conflict:
                p['enrichment_notes'] = ['来源字段已更新：本次旧补充暂停应用，请重新核对。'] + p['enrichment_notes']
                p['model_evidence_status'] = '补充与新来源冲突待核对'
            else:
                for key, value in changes.items():
                    if key != 'model_entries':
                        p[key] = value
                if 'model_entries' in changes:
                    rows = [dict(id=rid, model=e['model'], raw=e['raw'], tasks=e['tasks'],
                                 datasets=e['datasets'], scope='本次证据明确对应',
                                 source='data/enrichment.json', source_row=rid)
                            for e in changes['model_entries']]
                    p['model_evidence_status'] = changes.get('model_evidence_status', '本次正文补充')
        # Same definition as the former 524 metric, before role/name precision changes.
        legacy_models = [r['model'] for r in rows]
        p['legacy_named'] = any(named_record(m) for m in legacy_models)
        excluded = set(p['comparison_models'] + p['excluded_model_records'])
        rows = [r for r in rows if r['model'] not in excluded]
        if excluded and p['source'] == 'app_group_fulltext':
            p['datasets'] = sorted({d for r in rows for d in r['datasets']})
        p['models'] = sorted({r['model'] for r in rows})
        for row in rows:
            row['kind'] = model_kind(row['model'])
            row['evidence_status'] = p['model_evidence_status']
            final_models.append(row)
        p['qualified_models'] = [m for m in p['models'] if model_kind(m) == QUALIFIED]
        kinds = {model_kind(m) for m in p['models']}
        if p['qualified_models']:
            p['model_status'] = '模型／模型家族已记录'
        elif '仅工具／服务／预计算结果' in kinds:
            p['model_status'] = '仅工具／服务，架构待补'
        elif '骨干／方法描述，架构未完整说明' in kinds:
            p['model_status'] = '骨干／方法描述，架构待补'
        elif '传统视觉方法' in kinds:
            p['model_status'] = '传统视觉方法'
        elif p['comparison_models']:
            p['model_status'] = '仅比较基线具名'
        elif p['original_level'] == '仅摘要':
            p['model_status'] = '未获全文，摘要未具名'
        else:
            p['model_status'] = '现有全文文本不足以确认'
        if rid in audits:
            audit = dict(audits[rid])
            audit.update(title=p['title'], year=p['year'], theme=p['theme'], theme_detail=p['theme_detail'],
                         tasks=p['tasks'], models=p['models'], qualified_models=p['qualified_models'],
                         level=p['level'], model_status=p['model_status'],
                         model_evidence_status=p['model_evidence_status'],
                         comparison_models=p['comparison_models'], excluded_model_records=p['excluded_model_records'],
                         notes=p['enrichment_notes'], evidence_sources=p['enrichment_sources'])
            if u and 'model_entries' in u['changes']:
                audit['model_note'] = '已补充；依据与限定条件见补充说明。'
            elif p['comparison_models'] and not p['qualified_models']:
                audit['model_note'] = '具名的是比较基线，原来仅取主用模型的统计将其排除。'
            audit['task_status'] = ('任务仍未说明，需全文方法证据' if '任务未说明' in p['tasks'] else
                                    '已确认图像分割，但分割类型未说明' if '图像分割（类型未说明）' in p['tasks'] else
                                    '本次补充任务' if p['tasks'] != p['original_tasks'] else '沿用已提取任务')
            audit_rows.append(audit)
    catalog['models'] = final_models
    catalog['pairs'] = [dict(id=r['id'], model=r['model'], dataset=d, scope=r['scope'])
                        for r in final_models for d in r['datasets']]
    catalog['audit'] = audit_rows
    catalog['enrichment'] = {k: v for k, v in supplement.items() if k not in ('updates', 'audit')}
    catalog['schema_version'] = 2
    return catalog
