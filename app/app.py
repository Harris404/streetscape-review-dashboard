from __future__ import annotations

import csv
import html
import importlib.util
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import TYPE_NAMES, fingerprint, load_catalog
from enrichment import QUALIFIED, model_kind, named_record

st.set_page_config(page_title='街景研究 · 文献分析', page_icon='🏙️', layout='wide', initial_sidebar_state='auto')
# validated categorical palette (dataviz reference, light mode): slot order is the CVD-safety mechanism
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
TEAL, NAVY = BLUE, ORANGE  # legacy names: application / method
INK, INK2, MUTED, SURFACE, GRID = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb', '#ecebe7'
SEQ = [[0, '#f3f6fa'], [.15, '#cde2fb'], [.4, '#86b6ef'], [.65, '#3987e5'], [.85, '#1c5cab'], [1, '#0d366b']]  # one-hue blue ramp
GROUPS = ['应用类', '算法与方法类']
GROUP_COLOR = {'应用类': BLUE, '算法与方法类': ORANGE}
TYPE_COLOR = {'迁移与适配': AQUA, '算法开发': VIOLET, '数据集研究': YELLOW, '模型比较': MAGENTA}
TASK_COLOR = {'检测': VIOLET, '分割': AQUA, '其他': YELLOW}
import os
_BUNDLED = Path(__file__).resolve().parent.parent / 'project'
PROJECT = Path(os.environ.get('STREETSCAPE_PROJECT') or (_BUNDLED if _BUNDLED.exists() else '/Users/paris404/Desktop/streetscape_review'))
OFFICIAL = PROJECT / 'analysis/two_waves/papers_two_waves.csv'
WAVE2 = PROJECT / 'analysis/wave2'
if not OFFICIAL.exists():
    OFFICIAL = Path(__file__).resolve().parent / 'data/papers_two_waves.csv'
if not (WAVE2 / 'wave2_flat.csv').exists():
    WAVE2 = Path(__file__).resolve().parent / 'data/wave2'
st.markdown('''<style>
.stApp {background:#f6f6f4;} [data-testid="stSidebar"] {background:#efeeea;border-right:1px solid #e3e2dd;}
.block-container {max-width:1500px;padding-top:3.6rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;color:#0b0b0b;} h1 {font-weight:750 !important;font-size:2.1rem !important;}
h3 {font-size:1.15rem !important;margin-top:.6rem !important;} .eyebrow {font-size:11px;letter-spacing:2.2px;font-weight:700;color:#2a78d6;margin-bottom:8px;}
.subhead {color:#52514e;font-size:15px;line-height:1.75;margin-bottom:22px;max-width:1000px;}
.brand {font-weight:800;font-size:23px;letter-spacing:-1px;margin-bottom:2px;}
.brand small {font-size:10px;color:#718096;letter-spacing:2px;font-weight:600;display:block;margin:8px 0 18px;}
.badge {display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;color:white;margin-left:10px;vertical-align:middle;}
.def {background:#fcfcfb;border:1px solid #e3e2dd;border-radius:12px;padding:16px 20px;line-height:1.75;font-size:14px;color:#33465f;height:100%;}
.def b {color:#0b0b0b;} .def .tag {font-size:11px;font-weight:700;letter-spacing:1.5px;}
[data-testid="stMetric"] {background:#fcfcfb;border:1px solid #e3e2dd;border-radius:12px;padding:16px 20px;}
[data-testid="stMetricValue"] {font-size:1.9rem;color:#0b0b0b;} [data-testid="stMetricLabel"] {color:#52514e;}
[data-testid="stPlotlyChart"] {background:#fcfcfb;border:1px solid #e3e2dd;border-radius:12px;overflow:hidden;}
[data-testid="stDataFrame"] {border:1px solid #e2e9f0;border-radius:10px;}
div[data-testid="stExpander"] {background:white;}
.evidence {background:white;border:1px solid #e2e9f0;border-radius:12px;padding:18px 22px;line-height:1.8;}
.evidence .id {color:#2a78d6;font-size:12px;font-weight:700;letter-spacing:1px;}
.evidence h4 {font-size:17px;margin:8px 0 14px;}
.foot {color:#8290a2;font-size:12px;margin-top:25px;border-top:1px solid #e2e9f0;padding-top:14px;}
button[kind="primary"] {border-radius:8px;}
</style>''', unsafe_allow_html=True)


def file_version(*paths):
    return tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())


@st.cache_data(show_spinner=False)
def read_data(version):
    return load_catalog()


@st.cache_data(show_spinner=False)
def read_official(version):
    """Official list of the project (966): wave 1 = 应用类, wave 2 = 算法与方法类."""
    if not OFFICIAL.exists():
        return None
    with OFFICIAL.open(encoding='utf-8-sig', newline='') as f:
        return {r['rec_id']: r for r in csv.DictReader(f) if r['formal_included'].startswith('yes')}


@st.cache_data(show_spinner=False)
def read_wave2(version):
    flat = pd.read_csv(WAVE2 / 'wave2_flat.csv', encoding='utf-8-sig')
    spec = importlib.util.spec_from_file_location('gains_curated', WAVE2 / 'gains_curated.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    h2h = pd.read_csv(WAVE2 / 'wave2_h2h.csv', encoding='utf-8-sig').fillna('')
    from modelnames import unify
    flat['family'] = [unify(f, b) for f, b in zip(flat.family.fillna(''), flat.base_model.fillna(''))]
    h2h['families'] = h2h.families.apply(lambda fs: '; '.join(sorted({unify(f) for f in fs.split('; ') if f})))
    h2h['winner'] = h2h.winner.apply(lambda w: unify(w) if w else '')
    return flat, pd.DataFrame(mod.G, columns=mod.COLS), h2h


@st.cache_data(show_spinner=False)
def read_curves(version):
    import json as _json
    curves = _json.loads((WAVE2 / 'learning_curves.json').read_text(encoding='utf-8')) if (WAVE2 / 'learning_curves.json').exists() else []
    types = pd.read_csv(WAVE2 / 'note_types.csv', encoding='utf-8-sig') if (WAVE2 / 'note_types.csv').exists() else None
    return curves, types


catalog = read_data(fingerprint())
DATASET_XLSX = PROJECT / 'datasets/数据集表_v2.xlsx'


@st.cache_data(show_spinner=False)
def read_dataset_table(version):
    """Dataset specs read live from the project's master table (sheet '1 数据集总表')."""
    import openpyxl
    from data import dataset_name
    if not DATASET_XLSX.exists():
        return None
    wb = openpyxl.load_workbook(DATASET_XLSX, read_only=True, data_only=True)
    rows = list(wb['1 数据集总表'].values); head = rows[0]; out = []
    for n, row in enumerate(rows[1:], 2):
        r = dict(zip(head, row))
        if not r.get('数据集'):
            continue
        out.append(dict(name=dataset_name(r['数据集']), full_name=r['数据集'], year=r['年份'], reference=r['出处'], imagery=r['影像类型'],
                        scene=r['场景（室内/室外）'], size=r['规模与划分'], resolution=r['分辨率'], classes=r['类别数'],
                        annotation=r['标注类型'] or '', license=r['许可'], fit=r['与街景研究的贴合度'], geography=r['采集地区/背景'],
                        tasks=[], source_row=n))
    return out


_live_ds = read_dataset_table(file_version(DATASET_XLSX))
if _live_ds:
    catalog = dict(catalog, datasets=_live_ds)
official = read_official(file_version(OFFICIAL))
papers = pd.DataFrame(catalog['papers'])
if official:
    papers = papers[papers.id.isin(official)].copy()
    papers['group'] = papers.id.map(lambda i: '应用类' if official[i]['wave'] == '1' else '算法与方法类')
    papers['type_name'] = papers.id.map(lambda i: TYPE_NAMES[official[i]['type']])
    papers['reclassified'] = papers.id.map(lambda i: 'original type' in official[i]['wave_note'])
    MISSING = sorted(set(official) - set(papers.id))
else:
    papers['reclassified'] = papers.classification_note.str.contains('original type', na=False)
    MISSING = []
all_models = pd.DataFrame(catalog['models'])


def _retheme(row):
    """Scene-text detection was excluded from the review (2026-10-28); the snapshot's '场景文字检测与识别' theme was also
    triggered by words like 'texture'/'context'. Re-derive the technical theme of those papers from their tasks."""
    if row['group'] != '算法与方法类' or row['theme'] != '场景文字检测与识别':
        return row['theme']
    t = set(row['tasks']) - {'场景文字检测', '其他任务'}
    det, seg = '目标检测' in t, bool(t & {'语义分割', '实例分割', '全景分割'})
    return '检测与分割综合方法' if det and seg else '街景目标检测与定位' if det else '街景分割与场景解析'


papers['theme'] = papers.apply(_retheme, axis=1)
all_pairs = pd.DataFrame(catalog['pairs']).drop_duplicates(['id', 'model', 'dataset', 'role', 'model_role'])


def _model_norm():
    """Same model-name normalisation as the project pipeline (app_group/aggregate_app_ft.py RULES)."""
    import re as _re
    src = (PROJECT / 'app_group/aggregate_app_ft.py').read_text(encoding='utf-8')
    rules = eval(src[src.index('RULES = [') + 8: src.index(']\ndef norm') + 1])

    def norm(n):
        t = str(n).lower().replace('proposed:', '').strip()
        for k, v in rules:
            if (_re.search(k, t) if k.startswith('\\') else k in t):
                return v
        return str(n).strip()
    return norm


@st.cache_data(show_spinner=False)
def comparison_sources(version):
    """Base model per method paper (wave-2 extraction) and comparison models of application papers (app-group extraction)."""
    norm = _model_norm()
    flat = pd.read_csv(WAVE2 / 'wave2_flat.csv', encoding='utf-8-sig')
    base = {r.id: norm(r.base_model) for r in flat.itertuples() if isinstance(r.base_model, str)}
    with (PROJECT / 'app_group/app_group_models_long.csv').open(encoding='utf-8-sig', newline='') as f:
        comp = [r for r in csv.DictReader(f) if r['role'] == 'comparison']
    return base, comp


def split_roles(models, pairs, records):
    from modelnames import standardise
    flat = pd.read_csv(WAVE2 / 'wave2_flat.csv', encoding='utf-8-sig')
    _, comp = comparison_sources(file_version(WAVE2 / 'wave2_flat.csv', PROJECT / 'app_group/app_group_models_long.csv'))
    enet_ok = {i for i, r in (official or {}).items() if 'ENet' in [m.strip() for m in r['main_models'].split(';')]}
    enet_ok |= set(flat.loc[flat.family == 'ENet', 'id']) | {r['rec_id'] for r in comp if r['model'] == 'ENet'}
    return standardise(models, pairs, records, papers, flat, comp, _model_norm(), model_kind, enet_ok)


all_models, all_pairs, _records = split_roles(all_models, all_pairs, pd.DataFrame(catalog['dataset_records']))
catalog = dict(catalog, dataset_records=_records.to_dict('records'))
dataset_df = pd.DataFrame(catalog['datasets'])
coverage_df = pd.DataFrame(catalog['coverage'])
N_ALL = len(papers)


def clean_model(x):
    return model_kind(x) == QUALIFIED


def apply_filters(frame, years, chosen_tasks, chosen_themes, chosen_types, chosen_models, chosen_data, search):
    f = frame[frame.year.between(years[0], years[1])].copy()
    for key, choices in [('tasks', chosen_tasks), ('models', chosen_models), ('datasets', chosen_data)]:
        if choices:
            f = f[f[key].apply(lambda values: bool(set(values) & set(choices)))]
    if chosen_themes:
        f = f[f.theme.isin(chosen_themes)]
    if chosen_types:
        f = f[f.type_name.isin(chosen_types)]
    if search.strip():
        needle = search.strip().casefold()
        f = f[f.apply(lambda r: needle in (' '.join([r['id'], r['title'], r['location'], r['classes']])).casefold(), axis=1)]
    return f


def reset_filters():
    for key in ('years', 'tasks', 'themes', 'types', 'models', 'datasets', 'search'):
        st.session_state.pop(key, None)


def goto(view, page='研究概览'):
    def _go():
        reset_filters()
        st.session_state.view = view
        st.session_state.page = page
    return _go


VIEWS = ['研究全景'] + GROUPS
with st.sidebar:
    st.markdown('<div class="brand">街景研究<small>RESEARCH EXPLORER</small></div>', unsafe_allow_html=True)
    view = st.radio('视图', VIEWS, key='view', on_change=reset_filters,
                    format_func=lambda v: {'研究全景': '研究全景与数据集（总览）', '应用类': f'应用类 · {(papers.group == "应用类").sum()} 篇', '算法与方法类': f'算法与方法类 · {(papers.group == "算法与方法类").sum()} 篇'}[v])
    group = view if view in GROUPS else None
    selected = papers.iloc[0:0]
    page = view
    if group:
        PAGES = ['研究概览', '模型与数据来源'] + (['训练与迁移'] if group == '算法与方法类' else [])
        if st.session_state.get('page') not in PAGES:
            st.session_state.page = PAGES[0]
        st.divider()
        st.markdown(f'**{group} · 分析页面**')
        page = st.radio('分析页面', PAGES, label_visibility='collapsed', key='page')
        st.divider()
        scope = papers[papers.group == group].copy()
        N_SCOPE = len(scope)
        st.markdown(f'**{group}筛选**')
        years = st.slider('发表年份', int(scope.year.min()), int(scope.year.max()),
                          (int(scope.year.min()), int(scope.year.max())), key='years')
        chosen_tasks = st.multiselect('视觉任务', sorted({t for ts in scope.tasks for t in ts}), key='tasks', placeholder='全部任务')
        chosen_types = []
        if group == '算法与方法类':
            chosen_types = st.multiselect('子类', [t for t in ['迁移与适配', '算法开发', '数据集研究', '模型比较'] if t in set(scope.type_name)],
                                          key='types', placeholder='全部子类')
        chosen_themes = st.multiselect('研究主题' if group == '应用类' else '技术主题', sorted(scope.theme.unique()), key='themes', placeholder='全部主题')
        with st.expander('模型与数据筛选'):
            chosen_models = st.multiselect('使用模型', sorted({m for ms in scope.qualified_models for m in ms}), key='models', placeholder='全部模型')
            chosen_data = st.multiselect('训练数据来源', sorted({d for ds in scope.datasets for d in ds}), key='datasets', placeholder='全部数据集')
        search = st.text_input('搜索论文／ID／对象', placeholder='例如 tree、R0143', key='search')
        st.button('清空筛选', on_click=reset_filters, width='stretch', key='reset_filters')
        selected = apply_filters(scope, years, chosen_tasks, chosen_themes, chosen_types, chosen_models, chosen_data, search)
        st.caption(f'当前范围：{len(selected):,} / {N_SCOPE:,} 篇')
    else:
        st.caption(f'全部正式纳入 {N_ALL:,} 篇（两类共用页面）。选择“应用类”或“算法与方法类”进入各自的分析页面与筛选。')
    st.divider()
    st.caption('分类口径 · 项目正式清单（papers_two_waves.csv）' if official else '分类口径 · 应用内快照')
    if st.button('重新读取数据', width='stretch'):
        read_data.clear(); read_official.clear(); read_wave2.clear(); read_dataset_table.clear(); read_curves.clear()
        st.rerun()

ids = set(selected.id)


def hero(kicker, title, subtitle, badge=None):
    st.markdown(f'<div class="eyebrow">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    b = f'<span class="badge" style="background:{GROUP_COLOR[badge]}">{badge}</span>' if badge else ''
    st.markdown(f'<h1 style="margin:0 0 8px 0">{html.escape(title)}{b}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="subhead">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def metrics(items):
    for column, (label, value, note) in zip(st.columns(len(items)), items):
        with column:
            st.metric(label, value, help=note)


def format_fig(fig, height=360):
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=20, r=24, t=40, b=46),
                      font=dict(family='Arial, PingFang SC, Microsoft YaHei, sans-serif', size=12, color=INK2),
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=COLORS, barcornerradius=4, bargap=.28,
                      legend=dict(orientation='h', y=1.12, x=0, title=None), hoverlabel=dict(bgcolor='white'),
                      xaxis=dict(gridcolor=GRID, zeroline=False, automargin=True, linecolor=GRID),
                      yaxis=dict(gridcolor=GRID, zeroline=False, automargin=True, linecolor=GRID))
    return fig


def chart(fig, key, select=False, height=360):
    return st.plotly_chart(format_fig(fig, height), width='stretch', key=key, theme=None,
                           on_select='rerun' if select else 'ignore',
                           selection_mode='points', config={'displaylogo': False, 'scrollZoom': False,
                           'toImageButtonOptions': {'format': 'png', 'scale': 3}})


def bar(counts, key, color=TEAL, horizontal=True, selectable=False, height=380):
    data = counts.rename_axis('类别').reset_index(name='论文数')
    if data.empty:
        st.info('当前筛选下没有可统计记录。'); return None
    if horizontal:
        data = data.sort_values('论文数')
        fig = px.bar(data, y='类别', x='论文数', orientation='h', text='论文数', custom_data=['类别'])
        fig.update_layout(yaxis_title=None, xaxis_title='论文数（按 ID 去重）')
    else:
        fig = px.bar(data, x='类别', y='论文数', text='论文数', custom_data=['类别'])
        fig.update_layout(xaxis_title=None, yaxis_title='论文数（按 ID 去重）')
    fig.update_traces(marker_color=color, textposition='outside', cliponaxis=False,
                      hovertemplate='%{customdata[0]}<br>%{x} 篇<extra></extra>' if horizontal else None)
    return chart(fig, key, selectable, height)


def chosen_point(event):
    if event and event.selection.points:
        point = event.selection.points[0]
        return point.get('customdata', [None])[0]
    return None


def counts_for(frame, column):
    long = frame[['id', column]].explode(column)
    long = long[long[column].notna() & (long[column] != '')].drop_duplicates(['id', column])
    return long.groupby(column).id.nunique().sort_values(ascending=False)


def matrix(frame, row, col, key, height=430, percent=False, row_order=None, col_order=None, color=TEAL):
    if frame.empty:
        st.info('当前筛选下没有可建立对应关系的记录。'); return
    table = frame.drop_duplicates(['id', row, col]).groupby([row, col]).id.nunique().unstack(fill_value=0)
    if row_order is None:
        row_order = table.sum(axis=1).sort_values(ascending=False).index.tolist()
    if col_order is None:
        col_order = table.sum().sort_values(ascending=False).index.tolist()
    table = table.reindex(index=row_order, columns=col_order, fill_value=0)
    z = table.astype(float)
    if percent:
        z = z.div(z.sum(axis=0).replace(0, 1), axis=1) * 100
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in table.columns], y=table.index.tolist(),
        customdata=table.values, colorscale=SEQ,
        colorbar=dict(title='%' if percent else '篇'), text=z.round(1).values if percent else table.values,
        texttemplate='%{text:.1f}%' if percent else '%{text}',
        hovertemplate='%{y} × %{x}<br>%{customdata} 篇<extra></extra>'))
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_autorange='reversed')
    chart(fig, key, height=height)


def paper_evidence(frame, key, label='查看论文与证据', expanded=False):
    with st.expander(f'{label} · {len(frame):,} 篇', expanded=expanded):
        if frame.empty:
            st.info('没有符合条件的论文。'); return
        view = frame[['id', 'year', 'title', 'type_name', 'tasks', 'theme', 'model_variants', 'datasets']].copy()
        for c in ('tasks', 'model_variants', 'datasets'):
            view[c] = view[c].apply(lambda x: '；'.join(x))
        view.columns = ['论文 ID', '年份', '标题', '类型', '视觉任务', '主题', '模型', '训练数据来源']
        view = view.sort_values(['年份', '论文 ID'], ascending=[False, True])
        event = st.dataframe(view, hide_index=True, width='stretch', height=320,
                            on_select='rerun', selection_mode='single-row', key=f'{key}_table',
                            column_config={'标题': st.column_config.TextColumn(width='large')})
        st.download_button('下载当前论文清单 · CSV', view.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'{key}_papers.csv', mime='text/csv', key=f'{key}_download')
        selected_rows = event.selection.rows
        if selected_rows:
            rid = view.iloc[selected_rows[0]]['论文 ID']
        else:
            rid = st.selectbox('选择论文查看证据', view['论文 ID'].tolist(),
                               format_func=lambda v: v + ' · ' + frame.set_index('id').loc[v, 'title'][:75], key=f'{key}_paper')
        r = frame.set_index('id').loc[rid]
        st.markdown(f'<div class="evidence"><div class="id">{html.escape(rid)} · {html.escape(r["type_name"])}</div>'
                    f'<h4>{html.escape(r["title"])}</h4>{html.escape(r["evidence"] or "当前来源没有证据摘录。")}</div>', unsafe_allow_html=True)
        if r.get('reclassified'):
            st.caption('该论文原为算法／比较研究，未训练模型，直接使用现成或零样本模型，按“分类 = 是否训练”归入应用类。')
        a, b = st.columns(2)
        with a:
            st.caption('识别对象：' + (r['classes'] or '未单独提取'))
            st.caption('应用指标：' + (r['indicators'] or '未单独提取'))
        with b:
            st.caption('影像来源：' + (r['imagery'] or '未说明'))
            st.caption('框架／模型渠道：' + (r['framework'] or '未说明'))
        if r['doi']:
            st.link_button('打开论文 DOI', 'https://doi.org/' + r['doi'].removeprefix('https://doi.org/'))
        st.write('**模型记录状态**：' + r['model_status'])
        if r['theme_detail']:
            st.write('**主题细分**：' + r['theme_detail'])
        if r['comparison_models']:
            st.write('**比较基线（单列）**：' + '；'.join(r['comparison_models']))
        st.write('**训练数据状态**：' + r['dataset_status'])
        if r['dataset_records']:
            dr = pd.DataFrame(r['dataset_records']).replace({'role': {'pretrain': '预训练', 'train': '训练／微调', 'unspecified': '训练来源（角色未细分）', 'released': '新发布数据集'}})
            st.dataframe(dr.rename(columns={'name': '数据集', 'role': '数据用途', 'model': '对应模型', 'model_role': '模型角色'}), hide_index=True, width='stretch')
        if r['enrichment_notes'] or r['enrichment_sources']:
            st.markdown('**补充与复核**')
            for note in r['enrichment_notes']:
                st.write('• ' + note)
            st.dataframe(pd.DataFrame(r['enrichment_sources']).rename(columns={
                'field': '字段', 'source': '证据来源', 'locator': '位置', 'evidence': '证据／摘要'}),
                hide_index=True, width='stretch')


# ---------------------------------------------------------------- pages
def landscape():
    hero('01 / RESEARCH LANDSCAPE', '研究全景', '按作者是否训练模型分为两大类：直接使用现成模型的应用类，以及涉及训练或自建标注的算法与方法类。')
    app = papers[papers.group == '应用类']; meth = papers[papers.group == '算法与方法类']
    metrics([('正式纳入', f'{N_ALL:,}', '项目正式清单，按论文 ID 去重。'),
             ('应用类', f'{len(app):,}', '直接使用现成或零样本模型，含未训练模型的算法／比较研究。'),
             ('算法与方法类', f'{len(meth):,}', '涉及模型训练、微调或自建标注。')])
    prisma = catalog['prisma']
    st.caption(f'PRISMA：评估全文 {prisma["assessed"]} 篇，全文排除 {prisma["excluded"]} 篇，纳入 {prisma["included"]} 篇。纳入索引中的 {catalog["scope_sync"]["verified_files"]} 份PDF均已核实存在。')
    st.write('')
    a, b = st.columns(2, gap='large')
    with a:
        st.markdown(f'<div class="def"><span class="tag" style="color:{TEAL}">应用类 · {len(app)} 篇</span><br>'
                    '作者<b>不训练模型</b>，直接用现成的预训练或零样本模型（DeepLabv3+、PSPNet、SegFormer、SAM 等）识别街景要素，'
                    f'结果用于城市研究，或比较现成模型的效果。其中 {int(app.reclassified.sum())} 篇原为算法／比较研究。'
                    '<br>回答：<b>哪些模型拿来就能用？</b></div>', unsafe_allow_html=True)
    with b:
        sub = '、'.join(f'{k} {v}' for k, v in meth.type_name.value_counts().items())
        st.markdown(f'<div class="def"><span class="tag" style="color:{NAVY}">算法与方法类 · {len(meth)} 篇</span><br>'
                    f'作者<b>训练、微调或改进模型</b>，或自建街景标注数据集。子类：{sub}。'
                    '<br>回答：<b>哪些模型适合迁移、要标多少 label？</b></div>', unsafe_allow_html=True)
    st.write('')
    left, right = st.columns([1, 1.4], gap='large')
    with left:
        st.subheader('两大类占比')
        fig = go.Figure(go.Pie(labels=GROUPS, values=[len(app), len(meth)], hole=.68, marker=dict(colors=[BLUE, ORANGE], line=dict(color=SURFACE, width=2)),
                               textinfo='label+percent', textposition='outside', sort=False))
        fig.update_layout(showlegend=False, annotations=[dict(text=f'<b>{N_ALL}</b><br>篇', x=.5, y=.5, showarrow=False, font_size=22)])
        chart(fig, 'classification_pie', height=380)
    with right:
        st.subheader('算法与方法类的构成')
        bar(meth.type_name.value_counts(), 'method_types', color=ORANGE, height=380)
    st.subheader('两类研究的年度分布')
    trend = papers.groupby(['year', 'group']).id.nunique().reset_index(name='论文数')
    fig = px.bar(trend, x='year', y='论文数', color='group', barmode='stack', color_discrete_map=GROUP_COLOR,
                 labels={'year': '发表年份', 'group': '研究类型'}, category_orders={'group': GROUPS})
    fig.update_xaxes(dtick=1)
    chart(fig, 'all_years', height=340)
    cutoff = int(catalog['review_meta']['updated_at'][:4])
    future = int((papers.year > cutoff).sum())
    st.caption(f'按发表年份计数；{cutoff} 年为检索截面，不能与完整年度直接比较增速。'
               + (f' 另有 {future} 篇标为晚于检索年份，保留源清单年份。' if future else ''))
    st.caption('下方为两类共用的“数据集与对象覆盖”；各类的详细分析请在左侧选择“应用类”或“算法与方法类”。')
    if MISSING:
        st.caption(f'注意：正式清单中有 {len(MISSING)} 篇不在应用数据快照里，未显示：' + '、'.join(MISSING[:20]))



TYPE_DEF = [
    ('迁移与适配', '拿现成模型，在作者自己标注的街景数据上训练或微调，模型结构基本不改。',
     '例：用 YOLOv8 在自标的 1,000 张街景上训练，检测路面坑洞、电线杆。'),
    ('算法开发', '提出新模型或改进已有模型（改网络结构、损失函数、训练策略等），再与其他方法比较。',
     '例：在 DeepLabv3+ 上加注意力模块，提高街景分割精度。'),
    ('数据集研究', '主要贡献是发布新的街景检测／分割标注数据集，多数附带基线模型的成绩。',
     '例：标注街边设施、建筑立面或街头摊贩的公开数据集。'),
    ('模型比较', '作者自己训练多个现有模型，在同一份街景数据上比较谁更好。',
     '例：YOLOv5、Faster R-CNN、SSD 在同一份标注上对比精度和速度。'),
]


def method_intro():
    allm = papers[papers.group == '算法与方法类']
    cols = st.columns(4, gap='medium')
    for col, (name, what, eg) in zip(cols, TYPE_DEF):
        n = int((allm.type_name == name).sum())
        col.markdown(f'<div class="def"><span class="tag" style="color:{TYPE_COLOR[name]}">{name} · {n} 篇</span><br>'
                     f'{what}<br><span style="color:#8a8984;font-size:13px">{eg}</span></div>', unsafe_allow_html=True)
    st.write('')
    a, b = st.columns([3, 1], gap='medium')
    a.markdown('<div class="def"><span class="tag" style="color:#eb6834">“训练与迁移”页是什么</span><br>'
               '专门回答导师第二波的两个问题：<b>① 没有现成模型时，哪个基础模型迁移到街景上效果好；'
               '② 要自己标多少张图、多大的数据集效果才够。</b>数据来自对这 491 篇逐篇读全文的专项提取：'
               '作者用了什么基础模型、从什么权重开始训练、自标了多少张、训练前后成绩提升多少。</div>', unsafe_allow_html=True)
    b.button('打开训练与迁移 →', type='primary', on_click=lambda: st.session_state.update(page='训练与迁移'), key='open_training', width='stretch')
    st.write('')


def overview():
    color = GROUP_COLOR[group]
    if group == '应用类':
        hero('03 / APPLICATION STUDIES', '应用研究关注什么？', '从发表时间、视觉任务与研究主题出发，观察现成模型在城市研究中的用途。', group)
    else:
        hero('03 / METHOD STUDIES', '算法与方法研究在做什么？', '这一类论文的共同点：作者自己训练、微调或改进了模型，或者自己标注了街景数据。按论文的主要贡献分为 4 个子类。', group)
        method_intro()
    if selected.empty:
        st.info('当前条件没有匹配论文，请放宽左侧筛选。'); return
    known = selected.qualified_models.apply(bool).sum()
    if group == '应用类':
        metrics([('当前范围', f'{len(selected):,}', f'应用类共 {N_SCOPE} 篇。'),
                 ('主用模型明确', f'{known:,}', '有完整模型或模型家族描述的论文。'),
                 ('主题已编码', f'{(selected.theme != "主题未编码").sum():,}', '沿用逐篇核查编码，包括研究问题、结局与技术主题。'),
                 ('有训练来源记录', f'{selected.datasets.apply(bool).sum():,}', '模型的预训练／训练数据集已记录。')])
    else:
        tc = selected.type_name.value_counts()
        metrics([('当前范围', f'{len(selected):,}', f'算法与方法类共 {N_SCOPE} 篇。'),
                 ('迁移与适配', f'{tc.get("迁移与适配", 0):,}', '用现有模型在自标街景数据上训练或微调。'),
                 ('算法开发', f'{tc.get("算法开发", 0):,}', '提出新模型或改进模型。'),
                 ('数据集 + 比较', f'{tc.get("数据集研究", 0) + tc.get("模型比较", 0):,}', '发布街景数据集，或训练多个模型后比较。')])
    st.write('')
    left, right = st.columns([1.25, 1], gap='large')
    with left:
        st.subheader('年度趋势')
        if group == '应用类':
            annual = selected.groupby('year').id.nunique().reindex(range(years[0], years[1] + 1), fill_value=0)
            fig = px.bar(x=annual.index, y=annual.values, labels={'x': '发表年份', 'y': '论文数'}, text=annual.values)
            fig.update_traces(marker_color=color, textposition='outside')
        else:
            annual = selected.groupby(['year', 'type_name']).id.nunique().reset_index(name='论文数')
            fig = px.bar(annual, x='year', y='论文数', color='type_name', barmode='stack',
                         labels={'year': '发表年份', 'type_name': '子类'}, color_discrete_map=TYPE_COLOR)
        fig.update_xaxes(dtick=1)
        chart(fig, 'scope_years', height=380)
    with right:
        st.subheader('视觉任务构成')
        event = bar(counts_for(selected, 'tasks'), 'task_counts', color=color, selectable=True, height=380)
    st.caption('任务可多选，同一篇可同时做检测和分割，各任务篇数相加可能超过论文总数。点击柱子可查看对应论文。')
    task_selection = chosen_point(event)
    if task_selection:
        paper_evidence(selected[selected.tasks.apply(lambda ts: task_selection in ts)], 'task_drill', f'图中选择：{task_selection}', True)
    if group == '算法与方法类':
        st.subheader('子类 × 视觉任务')
        st_rows = selected[['id', 'type_name', 'tasks']].explode('tasks')
        matrix(st_rows, 'type_name', 'tasks', 'type_task', height=300, color=NAVY,
               row_order=[t for t in ['迁移与适配', '算法开发', '数据集研究', '模型比较'] if t in set(st_rows.type_name)])
    st.subheader('研究主题与变化' if group == '应用类' else '技术主题与变化')
    left, right = st.columns([1, 1.55], gap='large')
    with left:
        event_theme = bar(selected.theme.value_counts(), 'theme_counts', color=color, selectable=True, height=500)
    with right:
        mode = st.radio('年度主题图', ['论文数', '年内占比'], horizontal=True, key='theme_mode')
        matrix(selected[['id', 'year', 'theme']], 'theme', 'year', 'theme_years', height=455, color=color,
               percent=mode == '年内占比', col_order=list(range(years[0], years[1] + 1)))
    theme_selection = chosen_point(event_theme)
    if theme_selection:
        paper_evidence(selected[selected.theme == theme_selection], 'theme_drill', f'图中选择：{theme_selection}', True)
    paper_evidence(selected, 'overview')


def models_page():
    color = GROUP_COLOR[group]
    hero('04 / MODELS & DATA', '模型从哪里来，用于什么任务？',
         '模型家族、具体架构、训练来源与使用角色分开记录。' + ('应用类看“现成模型 + 预训练数据集”。' if group == '应用类' else '算法与方法类看“基础模型 + 初始化／训练数据”。'), group)
    if selected.empty:
        st.info('当前条件没有匹配论文。'); return
    role_choice = st.radio('模型使用角色', ['主用分割／检测模型', '主用＋实际运行的比较分支'], horizontal=True, key='model_role_scope')
    roles = ['main'] if role_choice.startswith('主用分割') else ['main', 'comparison']
    use_rows = all_models[all_models.id.isin(ids) & all_models.role.isin(roles)].copy()
    named_rows = use_rows[use_rows.kind == QUALIFIED]
    use_pairs = all_pairs[all_pairs.id.isin(ids) & all_pairs.model_role.isin(roles)].copy()
    st.caption('主用＝论文用来产出结果的模型（算法与方法类即作者训练/改进的基础模型）；比较分支＝作者自己也跑过、用来对比的其他模型（只在文献综述里提到的不算）。')
    known_keys = set(zip(named_rows.id, named_rows.model))
    relevant = use_pairs.loc[[(rid, m) in known_keys for rid, m in zip(use_pairs.id, use_pairs.model)]].copy()
    metrics([('当前论文', len(selected), '当前筛选范围。'),
             ('模型／家族明确', named_rows.id.nunique(), '按论文 ID 去重。'),
             ('有训练来源记录', selected.datasets.apply(bool).sum(), '包括公开训练集、自建训练集。'),
             ('模型—数据明确配对', relevant.id.nunique(), '同一模型与数据集有据对应，不做笛卡尔配对。')])
    left, right = st.columns([1.3, 1], gap='large')
    with left:
        st.subheader('模型使用频次')
        c1, c2 = st.columns(2)
        top_n = c1.slider('展示前 N 个模型', 5, 30, 15, key='model_top')
        name_scope = c2.radio('名称粒度', ['模型家族', '模型 + 骨干／规模'], horizontal=True, key='name_scope')
        model_col = 'model' if name_scope == '模型家族' else 'variant'
        mc = named_rows.groupby(model_col).id.nunique().sort_values(ascending=False)
        event = bar(mc.head(top_n), 'models_bar', color=color, selectable=True, height=520)
    with right:
        st.subheader('预训练数据集频次' if group == '应用类' else '训练／预训练数据来源频次')
        training_records = pd.DataFrame(catalog['dataset_records'])
        allowed_roles = ['main', 'paper'] if roles == ['main'] else ['main', 'comparison', 'paper']
        training_records = training_records[training_records.id.isin(ids) & training_records.model_role.isin(allowed_roles)]
        training_records = training_records[training_records.role.isin(['pretrain', 'train', 'unspecified'])]
        dc = training_records.groupby('name').id.nunique().sort_values(ascending=False)
        bar(dc.head(12), 'dataset_freq', color=INK2, height=600)
    picked = chosen_point(event)
    if picked:
        paper_evidence(selected[selected.id.isin(named_rows[named_rows[model_col] == picked].id)], 'model_drill', f'图中选择：{picked}', True)
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('模型 × 数据来源')
        top_models = relevant.groupby('model').id.nunique().sort_values(ascending=False).head(top_n).index.tolist()
        top_data = relevant.groupby('dataset').id.nunique().sort_values(ascending=False).head(8).index.tolist()
        matrix(relevant[relevant.model.isin(top_models) & relevant.dataset.isin(top_data)], 'model', 'dataset', 'model_data_heatmap',
               height=max(380, len(top_models) * 28 + 120), row_order=top_models, col_order=top_data, color=color)
    with right:
        st.subheader('模型 × 视觉任务')
        mt = named_rows[['id', 'model', 'tasks']].explode('tasks').dropna(subset=['tasks'])
        best = mt.groupby('model').id.nunique().sort_values(ascending=False).head(top_n).index.tolist()
        matrix(mt[mt.model.isin(best)], 'model', 'tasks', 'model_task_heatmap', height=max(380, len(best) * 28 + 120), row_order=best, color=color)
    st.caption('每格为去重论文数。基准表现、验证集、街景影像平台和辅助模型的数据集，不作为该模型的训练来源。')
    st.subheader('训练数据来源 × ' + ('研究主题' if group == '应用类' else '技术主题'))
    ds_papers = selected[['id', 'datasets', 'theme']].explode('datasets').dropna(subset=['datasets'])
    top_ds = ds_papers.groupby('datasets').id.nunique().sort_values(ascending=False).head(10).index.tolist()
    matrix(ds_papers[ds_papers.datasets.isin(top_ds)], 'theme', 'datasets', 'dataset_themes', height=520, col_order=top_ds)
    st.caption('论文使用的训练／预训练数据集 × 主题，按 ID 去重，随左侧筛选变化。数据集本身的规格与类别见“研究全景与数据集”页下半部分。')
    paper_evidence(selected, 'models')


def dataset_page():
    st.divider()
    hero('02 / DATASETS & OBJECTS', '数据集与对象覆盖', '两类研究共用：常用公开城市数据集支持什么任务、规模与采集背景如何、能识别哪些城市对象，以及按目标要素反查可用的类别体系。')
    tabs = st.tabs(['① 支持的任务', '② 数据集比较', '③ 对象与类别', '④ 按要素查找'])
    names = dataset_df.name.tolist()
    with tabs[0]:
        st.subheader('数据集本身支持什么视觉任务？')
        def native_tasks(d):  # from the 标注类型 column of 数据集表_v2.xlsx; '框' (boxes) = object detection
            if d['name'] == 'SA-1B':
                return ['无类别掩码分割']
            import re
            a = d['annotation'] or ''
            out = ['目标检测'] if re.search(r'检测|2D|(?<!3D )框', a) else []  # 2D boxes = detection; 3D-only boxes are not counted
            out += [label for w, label in [('语义', '语义分割'), ('实例', '实例分割'), ('全景', '全景分割')] if w in a]
            return out
        native = [{'数据集': d['name'], '视觉任务': t} for d in catalog['datasets'] for t in native_tasks(d)]
        table = pd.DataFrame(native).assign(value=1).pivot_table(index='数据集', columns='视觉任务', values='value', aggfunc='max', fill_value=0)
        fig = go.Figure(go.Heatmap(z=table.values, x=table.columns, y=table.index,
                         colorscale=[[0, '#f3f6fa'], [1, '#3987e5']], showscale=False,
                         text=table.replace({0: '—', 1: '●'}).values, texttemplate='%{text}',
                         hovertemplate='%{y} · %{x}<br>%{text}<extra></extra>'))
        chart(fig, 'native_tasks', height=440)
        st.caption('依据数据集表_v2.xlsx“1 数据集总表”的“标注类型”列整理：写了“框”或“检测”记为目标检测，写了语义／实例／全景记为对应分割。只描述表中收录的子集（如 IDD 只收了场景解析子集）。SA-1B 的掩码没有语义类别。')
    with tabs[1]:
        compare = st.multiselect('选择数据集进行比较', names, default=names, key='compare_ds')
        use_app = counts_for(papers[papers.group == '应用类'], 'datasets')
        use_meth = counts_for(papers[papers.group == '算法与方法类'], 'datasets')
        rows = []
        for d in catalog['datasets']:
            if d['name'] in compare:
                rows.append({'数据集': d['name'], '应用类论文数': int(use_app.get(d['name'], 0)), '算法与方法类论文数': int(use_meth.get(d['name'], 0)),
                             '支持的任务': '；'.join(native_tasks(d)), '标注类型（原表）': d['annotation'], '场景': d['scene'], '类别数／体系': d['classes'],
                             '规模与划分': d['size'], '采集背景': d['geography'], '与街景的贴合度': d['fit']})
        view = pd.DataFrame(rows)
        st.dataframe(view, hide_index=True, width='stretch', height=360)
        st.caption('论文数 = 该类论文中把此数据集作为训练／预训练来源的篇数（按 ID 去重）。')
        st.download_button('下载数据集比较 · CSV', view.to_csv(index=False).encode('utf-8-sig'), 'dataset_comparison.csv', 'text/csv')
        if compare:
            detail = st.selectbox('展开数据集详情', compare, key='dataset_detail')
            d = next(d for d in catalog['datasets'] if d['name'] == detail)
            st.markdown(f'**{d["full_name"]}** · {d["year"]}')
            for label, field in [('出处', 'reference'), ('影像类型', 'imagery'), ('分辨率', 'resolution'), ('许可记录', 'license')]:
                st.write(f'**{label}**：{d[field]}')
            st.caption(f'来源：数据集表_v2.xlsx · 1 数据集总表 · 第 {d["source_row"]} 行。')
    with tabs[2]:
        st.subheader('城市对象覆盖矩阵')
        variants = coverage_df.variant.drop_duplicates().tolist()
        chosen_variants = st.multiselect('比较类别体系／版本', variants,
             default=variants, key='coverage_variants')
        elements = coverage_df.element.drop_duplicates().tolist()
        chosen_elements = st.multiselect('只看部分对象（留空显示全部）', elements, key='coverage_elements')
        cov = coverage_df[coverage_df.variant.isin(chosen_variants)]
        if chosen_elements:
            cov = cov[cov.element.isin(chosen_elements)]
        if not cov.empty:
            c = cov.pivot(index='element', columns='variant', values='coverage').reindex(index=[e for e in elements if e in set(cov.element)], columns=chosen_variants)
            labels = cov.pivot(index='element', columns='variant', values='labels').reindex(index=c.index, columns=c.columns)
            fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=0, zmax=2,
                     customdata=labels.values, showscale=False,
                     colorscale=[[0, '#f0efec'], [.249, '#f0efec'], [.25, '#b7d3f6'], [.749, '#b7d3f6'], [.75, '#3987e5'], [1, '#3987e5']],
                     text=c.replace({0: '—', 1: '≈', 2: '✓'}).values, texttemplate='%{text}',
                     hovertemplate='%{y}<br>%{x}<br>%{customdata}<extra></extra>'))
            fig.update_layout(yaxis_autorange='reversed')
            chart(fig, 'coverage', height=max(400, len(c) * 26 + 130))
            st.download_button('下载当前覆盖矩阵 · CSV', labels.to_csv().encode('utf-8-sig'), 'object_coverage.csv', 'text/csv')
        st.caption('✓ 有独立类别　≈ 仅被更宽类别包含　— 没有对应类别。类别覆盖不代表模型的实际识别精度。')
        st.subheader('完整类别清单')
        ds = st.selectbox('查看哪个数据集的对象标签？', names, key='classlist_ds')
        for r in catalog['classlists']:
            if r['dataset'] == ds:
                with st.expander(f'{r["variant"]} · {r["subset"]} · {r["count"]} 类', expanded=True):
                    st.write(r['labels'])
    with tabs[3]:
        st.subheader('从目标对象反查类别体系')
        wanted = st.multiselect('希望识别哪些城市要素？', coverage_df.element.drop_duplicates().tolist(),
                               default=coverage_df.element.drop_duplicates().tolist(), key='wanted_objects')
        strict = st.toggle('只保留每个目标都有独立类别的体系', value=False, key='strict_coverage')
        if wanted:
            cov = coverage_df[coverage_df.element.isin(wanted)]
            summary = cov.groupby('variant').coverage.agg(['min', 'sum'])
            valid = summary.index[summary['min'] == 2] if strict else summary.sort_values('sum', ascending=False).index
            result = cov[cov.variant.isin(valid)].pivot(index='variant', columns='element', values='labels').reindex(columns=wanted)
            if result.empty:
                st.info('没有同时满足这些独立类别要求的体系。可关闭严格条件查看更宽类别。')
            else:
                st.dataframe(result, width='stretch')
            st.caption('只按现有类别表匹配；可用权重、目标地区表现和标注成本需另行核实。')


BINS = [(1, 100, '≤100'), (101, 300, '101–300'), (301, 1000, '301–1千'), (1001, 3000, '1千–3千'), (3001, 10000, '3千–1万'), (10001, 10**9, '>1万')]


def training_page():
    hero('05 / TRAINING & TRANSFER', '训练与迁移：要标多少、哪个模型好迁移？',
         '对应导师第二波的两个问题：① 要标多少 label、多大的数据集效果才够；② 没有现成模型时，哪个基础模型迁移过来效果好。数据来自算法与方法类逐篇全文专项提取。', group)
    st.markdown('<div class="def">“迁移”指把在大数据集（COCO、Cityscapes、ImageNet 等）上训练过的模型，拿到街景任务上继续训练或微调。'
                '本页按证据强弱排列：先看<b>同一篇论文里的直接对比</b>（最可靠），再看<b>实际标注规模与起点权重</b>（描述现状），'
                '最后看<b>哪些模型常用、常胜出</b>（参考）。</div>', unsafe_allow_html=True)
    st.write('')
    flat_all, gains_all, h2h_all = read_wave2(file_version(WAVE2 / 'wave2_flat.csv', WAVE2 / 'gains_curated.py', WAVE2 / 'wave2_h2h.csv'))
    curves_all, ntypes = read_curves(file_version(WAVE2 / 'learning_curves.json', WAVE2 / 'note_types.csv'))
    flat = flat_all[flat_all.id.isin(ids)].copy()
    gains = gains_all[gains_all.id.isin(ids)].copy()
    if flat.empty:
        st.info('当前筛选下没有专项提取记录。'); return
    own = flat[flat.data_source.isin(['own-annotated', 'both'])].copy()
    own['own_images'] = pd.to_numeric(own.own_images, errors='coerce')
    counted = own.dropna(subset=['own_images']).copy()
    g = gains.copy()
    g['标注图像'] = pd.to_numeric(g['适配用标注图像'], errors='coerce')
    g['提升'] = pd.to_numeric(g['提升(点)'], errors='coerce')
    ga = g[(g['类型'] == 'A') & g['标注图像'].gt(0) & g['提升'].notna()]
    curves = [c for c in curves_all if c['id'] in ids]

    # ---- conclusion cards
    st.markdown('### 本页结论')
    c = st.columns(4, gap='medium')
    card = lambda col, tag, body: col.markdown(f'<div class="def"><span class="tag" style="color:{ORANGE}">{tag}</span><br>{body}</div>', unsafe_allow_html=True)
    if len(ga):
        card(c[0], '少量本地标注就有用', f'{len(ga)} 对同篇对比：用目标城市自己的图微调，中位标注 <b>{ga["标注图像"].median():,.0f} 张</b>，中位提升 <b>{ga["提升"].median():.1f} 个百分点</b>（{ga["提升"].min():.0f}–{ga["提升"].max():.0f}）。')
    curve_ids = sorted({x['id'] for x in curves if x['type'] == 'curve'})
    flat_pl = [x for x in curves if x['type'] == 'curve' and any(k in (x.get('plateau_note') or '') for k in ('不再', '无提升', '无明显提升'))]
    flat_pl.sort(key=lambda x: (0 if '张' in x['plateau_note'] else 1, x['id']))
    rising = [x for x in curves if x['type'] == 'curve' and any(k in (x.get('plateau_note') or '') for k in ('仍在上升', '未见平台'))]
    body = f'{len(curve_ids)} 篇论文在同一篇里用不同标注量训练。'
    if flat_pl:
        body += '出现平台期：' + '；'.join(f'{x["id"]} {x["plateau_note"]}' for x in flat_pl[:3]) + '。'
    if rising:
        body += '也有到全量仍在上升的：' + '、'.join(x['id'] for x in rising[:3]) + '。'
    card(c[1], '收益递减（篇内证据）', body)
    if len(counted):
        q = counted.own_images.quantile([.25, .5, .75])
        med = counted.groupby('task_group').own_images.median()
        card(c[2], '实际标注规模', f'自标图像中位 <b>{q[.5]:,.0f} 张</b>（一半在 {q[.25]:,.0f}–{q[.75]:,.0f}）；检测 {med.get("检测", float("nan")):,.0f}、分割 {med.get("分割", float("nan")):,.0f}。这是“实际标了多少”，不是“至少要多少”。')
    ns = (flat.init == 'not stated').mean() * 100
    card(c[3], '从什么权重起步', f'检测多从 <b>COCO</b> 起步，分割多从 <b>ImageNet 骨干 / Cityscapes</b> 起步；<b>{ns:.0f}%</b> 的论文没写初始化权重。')
    st.write('')

    # ---- 1. before / after
    st.markdown('### ① 同一模型、同一测试集：用本地标注微调前后提升多少？')
    if gains.empty:
        st.info('当前筛选下没有适配前后对比。')
    else:
        left, right = st.columns([1, 1.25], gap='large')
        with left:
            if ga.empty:
                st.info('没有可作图的数值记录。')
            else:
                fig = px.scatter(ga, x='标注图像', y='提升', text='id', log_x=True, hover_data=['模型', '指标', '适配前', '适配后', '测试集'],
                                 labels={'标注图像': '用于微调的目标城市标注图像数（对数轴）', '提升': '提升（百分点）'})
                fig.update_traces(marker=dict(size=11, color=ORANGE, line=dict(color=SURFACE, width=2)), textposition='top center', textfont=dict(size=10, color=INK2))
                fig.add_hline(y=ga['提升'].median(), line_dash='dot', line_color=MUTED, annotation_text=f'中位提升 {ga["提升"].median():.1f}', annotation_font_color=INK2)
                chart(fig, 'gain_scatter', height=430)
                st.caption('每个点是一篇论文：同一模型在同一测试集上，加入目标城市自标图像微调前后的差值。几十到几百张就能带来明显提升。')
        with right:
            show = gains[['id', '类型', '模型', '指标', '适配前', '适配后', '提升(点)', '适配用标注图像']].copy()
            show['类型'] = show['类型'].map({'A': '本地自标微调', 'B': '无目标标签', 'C': '预训练/策略/清洗'})
            st.dataframe(show.astype(str), hide_index=True, width='stretch', height=430)
            st.caption('A＝用目标城市自标图像微调；B＝不用目标标签（无监督适应、合成数据等）；C＝换预训练来源、训练策略或清洗数据。均为人工逐条核对。')

    # ---- 2. learning curves
    st.markdown('### ② 篇内学习曲线：标得越多，提升会不会停？')
    if not curves:
        st.info('学习曲线数据尚未生成或当前筛选下没有。')
    else:
        rows = []
        multi = pd.Series([(x['id'], x['type'], x.get('x_unit', '')) for x in curves]).value_counts()
        for x in curves:
            ys = [float(pt.get('y', 0) or 0) for pt in x['points']]
            # one paper can report several sub-experiments (cities / datasets): draw each as its own line
            series = x['id'] if multi[(x['id'], x['type'], x.get('x_unit', ''))] == 1 else f"{x['id']} · {str(x.get('test_set', '')).split(',')[0][:18]}"
            scale = 100 if (str(x.get('metric_scale', '')).strip() == '0-1' or (ys and max(ys) <= 1)) else 1
            for pt in x['points']:
                try:
                    rows.append(dict(id=x['id'], 曲线=series, 类型=x['type'], 模型=x.get('model', ''), 指标=x.get('metric', ''), 单位=x.get('x_unit', ''),
                                     x=float(pt['x']), y=float(pt['y']) * scale, 标签=pt.get('label', ''), 平台=x.get('plateau_note', '')))
                except (TypeError, ValueError, KeyError):
                    pass
        lc = pd.DataFrame(rows)
        kind = st.radio('曲线类型', ['学习曲线（同篇不同标注量）', '加入本地数据前后'], horizontal=True, key='curve_kind')
        sub_all = lc[lc['类型'] == ('curve' if kind.startswith('学习') else 'local_add')]
        UNIT = {'images': '图像张数', 'percent': '训练数据比例 %', 'instances': '标注实例数', 'patches': '图块数', 'other': '其他（按配置顺序）'}

        def curve_chart(sub, unit, key):
            sub = sub[sub['单位'] == unit].sort_values(['曲线', 'x'])
            if sub.empty:
                st.info('该类型下没有记录。'); return
            logx = unit in ('images', 'instances', 'patches')
            sub = sub.assign(x=sub.x.where(~(logx & sub.x.eq(0)), 1))  # 0 (no local data) is drawn at 1 on the log axis
            fig = px.line(sub, x='x', y='y', color='曲线', markers=True, log_x=logx,
                          hover_data=['模型', '指标', '标签', '平台'], color_discrete_sequence=COLORS,
                          labels={'x': {'images': '标注图像数', 'other': '配置顺序'}.get(unit, UNIT[unit]), 'y': '成绩（统一为 0–100）', '曲线': '论文'})
            fig.update_traces(line=dict(width=2), marker=dict(size=8))
            chart(fig, key, height=460)

        st.markdown(f'**横轴：标注图像张数** · {sub_all[sub_all["单位"] == "images"]["曲线"].nunique()} 条曲线')
        curve_chart(sub_all, 'images', 'curves')
        st.caption('每条线是一篇论文（同一篇的不同城市／数据集分开画），指标各不相同，只看每条线自己的走势（是否变平），不要在线与线之间比高低。0–1 的指标已乘以 100；横轴为对数轴，“0 张（未加本地数据）”画在 1 的位置。')
        others = [u for u in ['instances', 'percent', 'patches', 'other'] if (sub_all['单位'] == u).any()]
        if others:
            with st.expander('其他横轴单位的曲线（标注实例数、训练数据比例等，' + '、'.join(f'{UNIT[u]} {sub_all[sub_all["单位"] == u]["曲线"].nunique()} 条' for u in others) + '）'):
                u = st.radio('横轴单位', others, horizontal=True, key='curve_unit', format_func=lambda x: UNIT[x])
                curve_chart(sub_all, u, 'curves_other')
                st.caption('这些论文没有按图像张数报告：实例数是标注的物体个数，比例是训练集的百分比，图块是从大图切出的小块，“其他”只是配置先后顺序。只看趋势，不能和上图的张数直接对应。')
    with st.expander('各论文关于“标注量与效果”的记录（按类型）', expanded=False):
        notes = flat[flat.label_efficiency_note.fillna('').str.len() > 30][['id', 'base_model', 'own_images', 'label_efficiency_note']].copy()
        if ntypes is not None:
            notes = notes.merge(ntypes, on='id', how='left')
            notes['type'] = notes['type'].map({'curve': '学习曲线', 'local_add': '加入本地数据', 'imbalance': '类别不平衡', 'qualitative': '定性说法'}).fillna('未分类')
            order = {'学习曲线': 0, '加入本地数据': 1, '类别不平衡': 2, '定性说法': 3, '未分类': 4}
            notes = notes.sort_values('type', key=lambda s: s.map(order))
            pick = st.multiselect('显示类型', list(order), default=['学习曲线', '加入本地数据'], key='note_types')
            notes = notes[notes['type'].isin(pick)]
            view = notes[['id', 'type', 'summary_zh', 'base_model', 'own_images', 'label_efficiency_note']]
            view.columns = ['论文 ID', '类型', '要点', '基础模型', '自标图像', '原文摘要（英文）']
        else:
            view = notes.rename(columns={'id': '论文 ID', 'base_model': '基础模型', 'own_images': '自标图像', 'label_efficiency_note': '原文摘要（英文）'})
        st.dataframe(view.astype(str), hide_index=True, width='stretch', height=420,
                     column_config={'原文摘要（英文）': st.column_config.TextColumn(width='large')})
        st.caption('“学习曲线”和“加入本地数据”两类有成对数字，可以当证据；“类别不平衡”提示要按类别保证样本；“定性说法”只作背景。')

    # ---- 3. labelling scale and starting weights
    st.markdown('### ③ 实际标注规模与起点权重（描述现状）')
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('作者自标了多少张街景图像？')
        counted['区间'] = counted.own_images.apply(lambda n: next(b[2] for b in BINS if b[0] <= n <= b[1]))
        dist = counted.groupby(['区间', 'task_group']).id.nunique().reset_index(name='论文数')
        fig = px.bar(dist, x='区间', y='论文数', color='task_group', barmode='stack',
                     category_orders={'区间': [b[2] for b in BINS], 'task_group': ['检测', '分割', '其他']},
                     color_discrete_map=TASK_COLOR, labels={'task_group': '任务'})
        chart(fig, 'own_dist', height=380)
        stats = counted.groupby('task_group').own_images.describe()[['count', '25%', '50%', '75%']].round(0)
        stats.columns = ['篇数', 'P25', '中位数', 'P75']; stats.index.name = '任务'
        st.dataframe(stats, width='stretch')
    with right:
        st.subheader('基础模型从什么权重开始训练？')
        fi = flat[['id', 'family', 'init']]
        top_f = fi.groupby('family').id.nunique().sort_values(ascending=False).head(12).index.tolist()
        top_i = fi.groupby('init').id.nunique().sort_values(ascending=False).head(7).index.tolist()
        matrix(fi[fi.family.isin(top_f) & fi.init.isin(top_i)], 'family', 'init', 'family_init', height=520,
               row_order=top_f, col_order=top_i, color=NAVY)
        st.caption('“not stated”表示论文没写从什么权重开始训练。')

    # ---- 4. which model families
    st.markdown('### ④ 哪些基础模型常用、常胜出（参考）')
    h2h = h2h_all[h2h_all.id.isin(ids)]
    rows = []
    for fam in sorted({f for fs in h2h.families for f in fs.split('; ') if f}):
        rows.append({'模型族': fam, '参与对比': int(h2h.families.apply(lambda fs: fam in fs.split('; ')).sum()), '被评最佳': int((h2h.winner == fam).sum())})
    hh = pd.DataFrame(rows)
    min_n = st.slider('至少参与几次对比', 2, 15, 5, key='h2h_min')
    hh = hh[hh['参与对比'] >= min_n].copy() if not hh.empty else hh
    if hh.empty:
        st.info('当前筛选下没有足够的同篇对比。')
    else:
        hh['胜率'] = (hh['被评最佳'] / hh['参与对比'] * 100).round(0)
        hh = hh.sort_values(['胜率', '参与对比'])
        fig = go.Figure()
        fig.add_bar(y=hh['模型族'], x=hh['参与对比'], orientation='h', name='参与对比', marker_color='#cde2fb', hovertemplate='%{y}<br>参与 %{x} 次<extra></extra>')
        fig.add_bar(y=hh['模型族'], x=hh['被评最佳'], orientation='h', name='被评最佳', marker_color=ORANGE,
                    text=[f'{w}/{a} · {r:.0f}%' for w, a, r in zip(hh['被评最佳'], hh['参与对比'], hh['胜率'])],
                    textposition='outside', textfont=dict(color=INK2), hovertemplate='%{y}<br>被评最佳 %{x} 次<extra></extra>')
        fig.update_layout(barmode='overlay', xaxis_title='论文数', yaxis_title=None)
        chart(fig, 'h2h_bar', height=max(380, len(hh) * 32 + 110))
    st.caption('同一篇里比较了 ≥2 个模型族时，各族记“参与”一次，作者评为最佳的记“胜出”一次。作者改进版计入基础族、作者倾向报告自己方法最好，所以胜率只作参考。')
    with st.expander('各模型族的最好成绩中位数（跨论文，证据弱）'):
        perf = flat.copy()
        for col in ('best_mAP50', 'best_mIoU'):
            perf[col] = pd.to_numeric(perf[col], errors='coerce')
        det = perf[perf.task_group == '检测'].groupby('family').agg(有mAP50篇数=('best_mAP50', 'count'), mAP50中位数=('best_mAP50', 'median'))
        seg = perf[perf.task_group == '分割'].groupby('family').agg(有mIoU篇数=('best_mIoU', 'count'), mIoU中位数=('best_mIoU', 'median'))
        fam = det.join(seg, how='outer').fillna({'有mAP50篇数': 0, '有mIoU篇数': 0})
        fam = fam[(fam['有mAP50篇数'] + fam['有mIoU篇数']) > 0]
        fam = fam.assign(合计=fam['有mAP50篇数'] + fam['有mIoU篇数']).sort_values('合计', ascending=False).head(15).round(1)
        fam.index.name = '模型族'
        st.dataframe(fam[['有mAP50篇数', 'mAP50中位数', '有mIoU篇数', 'mIoU中位数']], width='stretch',
                     column_config={'mAP50中位数': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f'),
                                    'mIoU中位数': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f')})
        st.caption('每篇取同一指标的最高值，再取中位数。各篇类别、测试集、难度不同，不能据此判断模型强弱。')
    with st.expander('跨论文：标注量 vs 成绩（为什么不能这样比）'):
        metric = st.radio('指标', ['检测 mAP50', '分割 mIoU'], horizontal=True, key='amount_metric')
        col, tg = ('best_mAP50', '检测') if metric.startswith('检测') else ('best_mIoU', '分割')
        sc = counted[(counted.task_group == tg)].copy()
        sc[col] = pd.to_numeric(sc[col], errors='coerce'); sc = sc.dropna(subset=[col])
        if not sc.empty:
            topf = sc.family.value_counts().head(6).index
            sc['模型族'] = sc.family.where(sc.family.isin(topf), '其他')
            fig = px.scatter(sc, x='own_images', y=col, color='模型族', log_x=True, hover_data=['id', 'base_model', 'n_classes'],
                             labels={'own_images': '作者自标图像数（对数轴）', col: metric.split()[1] + ' (%)'})
            fig.update_traces(marker=dict(size=9, opacity=.85, line=dict(color=SURFACE, width=1.5)))
            chart(fig, 'amount_effect', height=420)
        st.caption('点散成一片、看不出“标得越多分越高”，因为各篇类别数、测试集、难度都不同。所以“要标多少”的证据要看 ① 和 ②。')
    paper_evidence(selected, 'training')


if page == '研究全景':
    landscape()
    dataset_page()
    st.divider()
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button('进入应用类 →', type='primary', on_click=goto('应用类'), key='enter_app', width='stretch')
    c2.button('进入算法与方法类 →', on_click=goto('算法与方法类'), key='enter_method', width='stretch')
elif page == '研究概览':
    overview()
elif page == '模型与数据来源':
    models_page()
else:
    training_page()

st.markdown('<div class="foot">STREETSCAPE REVIEW · 按论文 ID 去重 · 支持筛选、图表选择、证据回查与 CSV 导出</div>', unsafe_allow_html=True)
with st.expander('数据来源与统计口径'):
    st.write('分类以项目正式清单为准：应用类 = 作者未训练模型、直接使用现成或零样本模型；算法与方法类 = 涉及训练、微调或自建标注。')
    st.write('模型名称已统一：“模型家族”把同一模型的不同写法合并（YOLO 按版本区分）；“模型 + 骨干／规模”只保留骨干网络或 YOLO 规模，去掉大小写与训练配置差异。算法与方法类的主用模型取自第二波逐篇全文提取。')
    st.write('“训练与迁移”页来自第二波全文专项提取（analysis/wave2）；适配前后对比为人工逐条核对。')
    st.write(catalog['review_meta']['scope'])
    st.caption('正式ID和分类已逐篇对齐两波统计_v3.xlsx与正式纳入966篇_索引.csv。PRISMA全文排除383篇，硬盘excluded目录193份PDF，两者分别记录。')
    st.caption('研究数量遵从正式清单；未确认的训练来源保留为空，不推测模型与数据集配对。此前补齐的5篇及其证据已包含在当前清单中。')
    st.caption('应用内证据快照更新：' + catalog['review_meta']['updated_at'])
