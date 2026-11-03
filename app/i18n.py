"""English display layer. Internal data and logic keep their original (Chinese) values; everything shown on screen goes
through tr() / tr_df(). Short category values are listed here; longer free text (evidence notes, dataset-table cells,
gains table, label-efficiency summaries) is in data/en.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

EN = {
    # classes and sub-types
    '应用类': 'Application', '算法与方法类': 'Algorithm & method', '研究全景': 'Landscape',
    '迁移与适配': 'Transfer & adaptation', '算法开发': 'Algorithm development', '数据集研究': 'Dataset study', '模型比较': 'Model comparison',
    # vision tasks
    '语义分割': 'Semantic segmentation', '目标检测': 'Object detection', '实例分割': 'Instance segmentation', '全景分割': 'Panoptic segmentation',
    '场景文字检测': 'Scene-text detection', '变化检测': 'Change detection', '其他任务': 'Other task', '任务未说明': 'Task not stated',
    '图像分割（类型未说明）': 'Image segmentation (type not stated)', '无类别掩码分割': 'Class-agnostic mask segmentation',
    '检测': 'Detection', '分割': 'Segmentation', '其他': 'Other',
    # application research themes
    '感知与美学': 'Perception & aesthetics', '健康与福祉': 'Health & well-being', '步行性与主动出行': 'Walkability & active travel',
    '交通与道路安全': 'Transport & road safety', '绿化与植被': 'Greenery & vegetation', '热环境与气候': 'Thermal environment & climate',
    '住房与经济': 'Housing & economy', '犯罪、失序与社会': 'Crime, disorder & social', '公平与环境正义': 'Equity & environmental justice',
    '土地利用与城市功能': 'Land use & urban function', '遗产与旅游': 'Heritage & tourism', '城市形态': 'Urban form & morphology',
    '环境质量': 'Environmental quality', '基础设施与资产': 'Infrastructure & assets', '其他主题': 'Other theme', '主题未编码': 'Theme not coded',
    '现成模型评测与工具': 'Off-the-shelf model evaluation & tools',
    # method technical themes
    '街景目标检测与定位': 'Street-view object detection & localisation', '街景分割与场景解析': 'Street-view segmentation & scene parsing',
    '检测与分割综合方法': 'Combined detection & segmentation', '街景数据集与标注体系': 'Street-view datasets & label schemes',
    '街景变化检测': 'Street-view change detection', '场景文字检测与识别': 'Scene-text detection & recognition',
    # model names produced by modelnames.unify
    'YOLO（版本未注明）': 'YOLO (version not stated)', 'DeepLab（版本未注明）': 'DeepLab (version not stated)', 'DETR 系列': 'DETR family',
    'Swin（其他检测/分割头）': 'Swin (other det./seg. head)', 'ViT（其他头）': 'ViT (other head)', '自建／通用 CNN': 'Custom / generic CNN',
    '传统机器学习（非深度）': 'Classical ML (non-deep)', '通用 CNN / 自建网络': 'Custom / generic CNN', 'classical ML / 非深度': 'Classical ML (non-deep)',
    '其他 / 未说明': 'Other / not stated', 'ResNeSt（分割头未报告）': 'ResNeSt (segmentation head not reported)',
    '未具名ADE20K分割模型': 'Unnamed ADE20K segmentation model', '未具名分割模型': 'Unnamed segmentation model',
    'Mask2Former（ZenSVI分割模块）': 'Mask2Former (ZenSVI segmentation module)',
    'DeepLabv3（LaDeco；正文骨干写作ResNet-269）': 'DeepLabv3 (LaDeco; backbone given as ResNet-269 in the text)',
    'Mask2Former（复用NUS Global Streetscapes预计算结果）': 'Mask2Former (reuses NUS Global Streetscapes precomputed results)',
    # model kinds and statuses
    '模型／模型家族': 'Model / model family', '骨干／方法描述，架构未完整说明': 'Backbone / method description, architecture incomplete',
    '仅工具／服务／预计算结果': 'Tool / service / precomputed results only', '传统视觉方法': 'Classical vision method', '未具名': 'Unnamed',
    '架构未完整报告': 'Architecture not fully reported', '作者提出的流程（非现成模型）': "Authors' own pipeline (not a ready-made model)",
    '模型／模型家族已记录': 'Model / family recorded', '不适用：数据集发布，未开展模型实验': 'N/A: dataset release, no model experiments',
    '仅比较分支使用分割／检测模型': 'Detection/segmentation model used only in a comparison branch',
    '已记录训练来源': 'Training source recorded', '全文检索未确认训练来源': 'Training source not confirmed in full text',
    '不适用：数据集发布': 'N/A: dataset release', '训练来源未确认': 'Training source not confirmed',
    '发表版未明确训练集名称': 'Published version does not name the training set',
    '方法类主题使用技术主题，不混入应用类15类研究结局。': 'Method studies use technical themes, not the 15 application research outcomes.',
    '沿用既有技术主题编码；已按966篇源清单归入第一波应用类。': 'Existing technical theme kept; placed in wave 1 (application) per the 966-paper list.',
    # dataset names
    '作者自建／自标数据': "Authors' own / self-annotated data", 'Bing门图像': 'Bing door images', 'Boston街景数据集': 'Boston street-view dataset',
    'Google Street View图像': 'Google Street View images', 'Mapillary街景图像': 'Mapillary street-view images',
    '自建：200张胡同全景标注图像': 'Own: 200 annotated hutong panoramas', '自建：台湾便利店招牌与交通信号灯': 'Own: Taiwan convenience-store signs and traffic lights',
    '自建：1500张GSV建筑平面图像': 'Own: 1,500 GSV building images', '自建：384张Mapillary图像': 'Own: 384 Mapillary images',
    '自建：Surabaya街景图像': 'Own: Surabaya street-view images', '自建：GSV建筑立面图像': 'Own: GSV building facade images',
    '自建：EFF与Mapillary Vistas摄像头图像': 'Own: EFF and Mapillary Vistas camera images', '自建：约3000张GSV电信资产标注': 'Own: ~3,000 annotated GSV telecom-asset images',
    '自建：虚拟街景数据集': 'Own: virtual street-view dataset', '自建：Hanoi 1217张店招实例分割图像': 'Own: 1,217 Hanoi shop-sign instance-segmentation images',
    '自建：188张广告牌图像': 'Own: 188 billboard images', '自建：百度街景22类建筑风格图像': 'Own: Baidu street-view images, 22 architectural styles',
    '自采：Ricoh Theta街景': 'Own capture: Ricoh Theta street views', '自建：500张水上全景语义标注图像': 'Own: 500 semantically annotated waterway panoramas',
    '自建：1000张人工标注街景图像': 'Own: 1,000 manually annotated street-view images', '自建：800张训练、100张验证的街景图像': 'Own: street-view images, 800 train / 100 val',
    '自建Atlanta街景与网络图片（Koo等2022）': 'Own: Atlanta street views and web images (Koo et al. 2022)', '自建目标城市街景集（Hu）': 'Own: target-city street-view set (Hu)',
    '自建鱼眼天空标注集（300幅，Xia）': 'Own: annotated fisheye sky set (300 images, Xia)', '自采街景视频': 'Own capture: street-view video',
    # theme details (application papers)
    '绿化感知距离与心理恢复': 'Perceived greenery distance and psychological restoration', '乡村空间绅士化': 'Rural gentrification',
    '路面病害与养护评估': 'Pavement distress and maintenance assessment', '人行道宽度清查': 'Sidewalk width inventory',
    '街道垃圾暴露的社会空间不平等': 'Socio-spatial inequality in street litter exposure', '城市更新类型与街景变化': 'Urban renewal types and street-view change',
    '感知安全与客观犯罪风险的错配': 'Mismatch between perceived safety and objective crime risk', '老年人户外跌倒风险': 'Outdoor fall risk of older adults',
    '建筑立面与屋顶绿化潜力': 'Facade and rooftop greening potential', '共享自行车路线偏好': 'Bike-share route preference',
    '街道植被层次与组成': 'Street vegetation layers and composition', '建筑色彩组织与视觉感知': 'Building colour organisation and visual perception',
    '历史商业街区界面与地方文化': 'Historic commercial street frontage and local culture', '街景愉悦与不愉悦感': 'Pleasant and unpleasant street scenes',
    '街道绿色品质评估': 'Street greenery quality assessment', '建筑用途识别与制图': 'Building use recognition and mapping',
    '城市街景感知分类': 'Urban street-scene perception classification', '首层高度与洪涝风险': 'First-floor height and flood risk',
    '街道微观设施与材料审计': 'Street micro-features and materials audit', '空气污染暴露建模': 'Air pollution exposure modelling',
    '街头摊贩经营区空间优化': 'Spatial optimisation of street-vending areas', '树木与建筑遮阴和日照模拟': 'Tree and building shade and sunlight simulation',
    '居住与行人绿化暴露不平等': 'Inequality in residential and pedestrian greenery exposure', '6G室外通信绕射损耗': '6G outdoor communication diffraction loss',
    '绿视率测量与人工方法核对': 'Green View Index measurement checked against manual methods', '老年行人与适老步行性': 'Older pedestrians and age-friendly walkability',
    '街景视觉品质预测': 'Street-view visual quality prediction', '历史街区建筑色彩和谐度': 'Building colour harmony in historic districts',
    '多模态街道观测数据及多用途示范': 'Multimodal street observation data and multi-purpose demonstration', '环境因素与分年龄行人数量': 'Environmental factors and pedestrian counts by age',
    '店招定位与POI生成': 'Shop-sign localisation and POI generation', '城市物理失序解释': 'Explaining urban physical disorder',
    '建筑高度与三维城市模型': 'Building height and 3D city models', '城市立面色彩制图': 'Urban facade colour mapping',
    '非正规城市区域与多用途城市制图': 'Informal urban areas and multi-purpose urban mapping', '不同城市形态中的行人视角绿化': 'Pedestrian-view greenery across urban forms',
    '不健康广告暴露': 'Exposure to unhealthy advertising', '道路提取与宽度估算': 'Road extraction and width estimation',
    '贫困和生计指标预测': 'Predicting poverty and livelihood indicators', '视听环境与人类感知': 'Audio-visual environment and human perception',
    '街道绿化质量与全景绿视率': 'Street greenery quality and panoramic Green View Index', '街道交通负荷': 'Street traffic load',
    '自然监视与社区安全': 'Natural surveillance and neighbourhood safety', '监控摄像头与种族多样性': 'Surveillance cameras and racial diversity',
    '消费空间的性别差异与均衡': 'Gender differences and balance in consumption spaces', '城中村识别': 'Urban village identification',
    '最低楼层高程与洪灾风险': 'Lowest floor elevation and flood risk', '立面光伏部署与发电潜力': 'Facade photovoltaic deployment and generation potential',
    '来源字段已更新：本次旧补充暂停应用，请重新核对。': 'Source fields updated: the older supplement is paused; please re-check.',
    # evidence-source field names
    '模型检索定位': 'Model location in text', '方法检索定位': 'Method location in text', '主题编码依据（研究问题）': 'Theme coding basis (research question)',
    '模型／任务': 'Model / task', '数据来源': 'Data source', '具体模型／预训练来源': 'Specific model / pretraining source',
    '模型架构／训练来源深度复核': 'Architecture / training-source deep review', '补充材料': 'Supplementary material',
    '具体模型／训练来源': 'Specific model / training source', '缺项依据': 'Basis for missing item', '训练来源': 'Training source', '来源检查': 'Source check',
    # learning-curve plateau notes
    '数据平衡比数量更重要': 'class balance matters more than amount', '扩充后无提升': 'no gain after adding data', '全量前仍在上升': 'still rising before full set',
    '5轮后无明显提升': 'no clear gain after round 5', '未见平台': 'no plateau seen', '约5千张后不再提升': 'no further gain after ~5,000 images',
    '约2千张后不再提升': 'no further gain after ~2,000 images',
    # coverage-matrix cell labels
    '≈ vegetation（树、灌木、绿篱合并，无法区分乔木与草地）': '≈ vegetation (trees, shrubs and hedges merged; trees cannot be told from grass)',
    '≈ Vegetation（不分乔木/灌木）': '≈ Vegetation (trees/shrubs not separated)', '≈ terrain（草地与裸土合并）': '≈ terrain (grass and bare soil merged)',
    '≈ Terrain（含草地）': '≈ Terrain (includes grass)', '✓ terrain（含草地）': '✓ terrain (includes grass)', '≈ pole（路灯杆计入 pole）': '≈ pole (streetlight poles counted as pole)',
    # dataset-table short cells
    '室外·城市街道（驾驶视角）': 'Outdoor · urban streets (driving view)', '语义、实例、全景分割（像素级）': 'Semantic, instance, panoptic segmentation (pixel-level)',
    '室内 + 室外混合：官方逐图场景标签中 street 是最多的单一场景（2,241 张，约 10%）；可归类图像中室内约 55%、室外约 45%（本综述按 Places365 室内外标记统计，非作者报告）':
        'Mixed indoor + outdoor: "street" is the most common single scene in the official per-image labels (2,241 images, ~10%); of classifiable images ~55% indoor, ~45% outdoor (counted by this review from Places365 indoor/outdoor tags, not reported by the authors)',
    '不固定（短边常缩放到 512）': 'Varies (short side often resized to 512)', '语义分割（另有实例、部件标注）': 'Semantic segmentation (also instance and part labels)',
    '室外·街道（多视角、多天气、多季节、昼夜）': 'Outdoor · streets (multiple viewpoints, weather, seasons, day/night)', '≥1920×1080，尺寸不一': '≥1920×1080, sizes vary',
    '语义、实例、全景分割': 'Semantic, instance, panoptic segmentation', 'Lin et al., ECCV 2014；Kirillov et al., CVPR 2019（全景）': 'Lin et al., ECCV 2014; Kirillov et al., CVPR 2019 (panoptic)',
    '室内 + 室外混合': 'Mixed indoor + outdoor', '不固定（~640×480）': 'Varies (~640×480)', '框、实例掩膜、全景': 'Boxes, instance masks, panoptic',
    '室外·城市道路（驾驶视角）': 'Outdoor · urban roads (driving view)', '室外·城市/乡村/高速（驾驶视角）': 'Outdoor · urban/rural/highway (driving view)',
    '约 1242×375': '~1242×375', '2D/3D 框、语义与实例分割': '2D/3D boxes, semantic and instance segmentation',
    '室外·城市道路（多天气、昼夜）': 'Outdoor · urban roads (multiple weather, day/night)',
    '框、语义/实例/全景分割、可行驶区域、车道线': 'Boxes, semantic/instance/panoptic segmentation, drivable area, lane markings',
    '不固定（~500×375）': 'Varies (~500×375)', '框、语义/实例分割': 'Boxes, semantic/instance segmentation', '室内 + 室外通用': 'General indoor + outdoor',
    '原图约 3300×4950，发布版短边 1500': 'Originals ~3300×4950; released version short side 1500', '实例级掩码（无语义标签）': 'Instance-level masks (no semantic labels)',
    '未核实': 'Not verified', '目标检测框': 'Object detection boxes', '语义、实例分割': 'Semantic, instance segmentation',
    '室外·非结构化道路（驾驶视角）': 'Outdoor · unstructured roads (driving view)', '多为 1080p（平均 1678×968）': 'Mostly 1080p (mean 1678×968)',
    '语义分割（含层次标签）': 'Semantic segmentation (hierarchical labels)',
    # class-list subsets
    '语义分割评测 19 类': 'Semantic segmentation, 19 evaluation classes', '实例分割 8 类': 'Instance segmentation, 8 classes',
    'SceneParsing 150 类（按像素频率排序）': 'SceneParsing 150 classes (by pixel frequency)', '65 类评测（另有 Unlabeled）': '65 evaluation classes (plus Unlabeled)',
    '检测/实例 80 类（thing）': 'Detection/instance 80 classes (thing)', '全景 53 类 stuff（与 80 thing 合计 133）': 'Panoptic 53 stuff classes (133 with the 80 things)',
    '常用 11 类': 'Common 11 classes', '2D 检测': '2D detection', '检测 10 类': 'Detection, 10 classes', '20 类': '20 classes',
    '124 标签（含 Unlabeled），MMSegmentation 名称': '124 labels (incl. Unlabeled), MMSegmentation names',
    '完整 32 类（官方 label_colors.txt）': 'Full 32 classes (official label_colors.txt)', '22 类参与评测（README 原拼写）': '22 evaluated classes (README spelling)',
    'level-1 7 类': 'level-1, 7 classes', 'level-4 30 类': 'level-4, 30 classes', '150 类中的 stuff 35 类': '35 stuff classes of the 150',
    'stuff 91 类': '91 stuff classes', '365 类（官网名称，含原拼写）': '365 classes (official names and spelling)', '无类别（class-agnostic）': 'No classes (class-agnostic)',
    # dataset-record roles and model roles
    '预训练': 'Pretraining', '训练／微调': 'Training / fine-tuning', '训练来源（角色未细分）': 'Training source (role not split)', '新发布数据集': 'Newly released dataset',
    'main': 'Main', 'comparison': 'Comparison', 'paper': 'Paper level', 'auxiliary': 'Auxiliary',
    # gains-table type codes
    '本地自标微调': 'Local self-annotated fine-tuning', '无目标标签': 'No target labels', '预训练/策略/清洗': 'Pretraining / strategy / cleaning',
    # note types
    '学习曲线': 'Learning curve', '加入本地数据': 'Adding local data', '类别不平衡': 'Class imbalance', '定性说法': 'Qualitative statement', '未分类': 'Unclassified',
    # evidence levels
    'PDF全文': 'PDF full text', '发表版全文': 'Published full text', '全文': 'Full text', '仅摘要': 'Abstract only',
}
_FREE = Path(__file__).resolve().parent / 'data' / 'en.json'
if _FREE.exists():
    EN.update(json.loads(_FREE.read_text(encoding='utf-8')))

_ZH = re.compile(r'[一-鿿]')
_RULES = [
    (re.compile(r'^PDF第(\d+)页；提取文本第(\d+)行$'), r'PDF p. \1; extracted text line \2'),
    (re.compile(r'^OCR文本第(\d+)行（页码未对齐）$'), r'OCR text line \1 (page not aligned)'),
]


def tr(x):
    """English for a displayed value; unknown values are returned unchanged."""
    if not isinstance(x, str) or not _ZH.search(x):
        return x
    if x in EN:
        return EN[x]
    for rx, rep in _RULES:
        if rx.match(x):
            return rx.sub(rep, x)
    if x.startswith(('自建', '自采')):
        return EN['作者自建／自标数据'] + ' — ' + x.split('：', 1)[-1] if '：' in x else EN['作者自建／自标数据']
    for sep, out in (('; ', '; '), ('；', '; '), ('、', ', ')):
        if sep in x:
            return out.join(tr(p) for p in x.split(sep))
    for k in _FAMILIES:  # standardised variants such as 'YOLO（版本未注明） (DarkNet-53)'
        if k in x:
            return x.replace(k, EN[k])
    return x


_FAMILIES = ['YOLO（版本未注明）', 'DeepLab（版本未注明）', 'DETR 系列', 'Swin（其他检测/分割头）', 'ViT（其他头）', '自建／通用 CNN',
             '传统机器学习（非深度）', 'ResNeSt（分割头未报告）']


def tr_list(xs):
    return [tr(x) for x in xs] if isinstance(xs, (list, tuple)) else tr(xs)


def tr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Translate every displayed cell (strings and list cells) and the index."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: tr_list(v) if isinstance(v, (list, tuple)) else tr(v))
    out.index = [tr(i) for i in out.index] if out.index.dtype == object else out.index
    out.columns = [tr(c) for c in out.columns]
    return out
