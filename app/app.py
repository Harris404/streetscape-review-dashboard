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
from i18n import tr, tr_df, tr_list

st.set_page_config(page_title='Streetscape Review · Literature Explorer', page_icon='🏙️', layout='wide', initial_sidebar_state='auto')
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
# the same colours keyed by the displayed (English) names
GROUP_COLOR_EN, TYPE_COLOR_EN, TASK_COLOR_EN = ({tr(k): v for k, v in m.items()} for m in (GROUP_COLOR, TYPE_COLOR, TASK_COLOR))
PAGE_EN = {'研究概览': 'Overview', '模型与数据来源': 'Models & data sources', '训练与迁移': 'Training & transfer'}
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
from modelnames import fix_paper_fields, fix_dataset_names
papers = fix_paper_fields(papers)
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
    import json as _json
    rawp = PROJECT / 'analysis/two_waves/raw_models.json'
    raw_models = _json.loads(rawp.read_text(encoding='utf-8')) if rawp.exists() else None
    off_models = {i: [m.strip() for m in r['main_models'].split(';') if m.strip()] for i, r in (official or {}).items()}
    return standardise(models, pairs, records, papers, flat, comp, _model_norm(), model_kind, enet_ok, raw_models, off_models)


all_models, all_pairs, _records = split_roles(all_models, all_pairs, pd.DataFrame(catalog['dataset_records']))
all_pairs = fix_dataset_names(all_pairs, 'dataset'); _records = fix_dataset_names(_records, 'name')
# per-paper model fields (sidebar filter, overview metric, evidence table) follow the standardised names
_main = all_models[(all_models.role == 'main') & (all_models.kind == QUALIFIED)]
_fam = _main.groupby('id').model.apply(lambda x: sorted(set(x))).to_dict()
_var = all_models[all_models.role.isin(['main', 'comparison'])].groupby('id').variant.apply(lambda x: sorted(set(map(str, x)))).to_dict()
papers['qualified_models'] = papers.id.map(lambda i: _fam.get(i, []))
papers['models'] = papers['qualified_models']
papers['model_variants'] = papers.id.map(lambda i: _var.get(i, []))
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
    st.markdown('<div class="brand">Streetscape<small>RESEARCH EXPLORER</small></div>', unsafe_allow_html=True)
    view = st.radio('View', VIEWS, key='view', on_change=reset_filters,
                    format_func=lambda v: {'研究全景': 'Landscape & datasets (overview)', '应用类': f'Application studies · {(papers.group == "应用类").sum()}', '算法与方法类': f'Algorithm & method studies · {(papers.group == "算法与方法类").sum()}'}[v])
    group = view if view in GROUPS else None
    selected = papers.iloc[0:0]
    page = view
    if group:
        PAGES = ['研究概览', '模型与数据来源'] + (['训练与迁移'] if group == '算法与方法类' else [])
        if st.session_state.get('page') not in PAGES:
            st.session_state.page = PAGES[0]
        st.divider()
        st.markdown(f'**{tr(group)} studies · pages**')
        page = st.radio('Page', PAGES, label_visibility='collapsed', key='page', format_func=PAGE_EN.get)
        st.divider()
        scope = papers[papers.group == group].copy()
        N_SCOPE = len(scope)
        st.markdown(f'**Filter {tr(group).lower()} studies**')
        years = st.slider('Publication year', int(scope.year.min()), int(scope.year.max()),
                          (int(scope.year.min()), int(scope.year.max())), key='years')
        chosen_tasks = st.multiselect('Vision task', sorted({t for ts in scope.tasks for t in ts}, key=tr), key='tasks', placeholder='All tasks', format_func=tr)
        chosen_types = []
        if group == '算法与方法类':
            chosen_types = st.multiselect('Sub-type', [t for t in ['迁移与适配', '算法开发', '数据集研究', '模型比较'] if t in set(scope.type_name)],
                                          key='types', placeholder='All sub-types', format_func=tr)
        chosen_themes = st.multiselect('Research theme' if group == '应用类' else 'Technical theme', sorted(scope.theme.unique(), key=tr), key='themes', placeholder='All themes', format_func=tr)
        with st.expander('Model & data filters'):
            chosen_models = st.multiselect('Model used', sorted({m for ms in scope.qualified_models for m in ms}, key=lambda m: str(tr(m)).lower()), key='models', placeholder='All models', format_func=tr)
            chosen_data = st.multiselect('Training data source', sorted({d for ds in scope.datasets for d in ds}, key=tr), key='datasets', placeholder='All datasets', format_func=tr)
        search = st.text_input('Search title / ID / objects', placeholder='e.g. tree, R0143', key='search')
        st.button('Clear filters', on_click=reset_filters, width='stretch', key='reset_filters')
        selected = apply_filters(scope, years, chosen_tasks, chosen_themes, chosen_types, chosen_models, chosen_data, search)
        st.caption(f'In scope: {len(selected):,} / {N_SCOPE:,} papers')
    else:
        st.caption(f'All {N_ALL:,} included papers (shared page for both classes). Choose "Application" or "Algorithm & method" for their own pages and filters.')
    st.divider()
    st.caption('Classification · official project list (papers_two_waves.csv)' if official else 'Classification · in-app snapshot')
    if st.button('Reload data', width='stretch'):
        read_data.clear(); read_official.clear(); read_wave2.clear(); read_dataset_table.clear(); read_curves.clear()
        st.rerun()

ids = set(selected.id)


def hero(kicker, title, subtitle, badge=None):
    st.markdown(f'<div class="eyebrow">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    b = f'<span class="badge" style="background:{GROUP_COLOR[badge]}">{tr(badge)}</span>' if badge else ''
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
    data = counts.rename_axis('key').reset_index(name='Papers')
    if data.empty:
        st.info('No records under the current filters.'); return None
    data['Category'] = data.key.map(tr)  # custom_data keeps the internal key so a click maps back to the data
    if horizontal:
        data = data.sort_values('Papers')
        fig = px.bar(data, y='Category', x='Papers', orientation='h', text='Papers', custom_data=['key', 'Category'])
        fig.update_layout(yaxis_title=None, xaxis_title='Papers (unique IDs)')
    else:
        fig = px.bar(data, x='Category', y='Papers', text='Papers', custom_data=['key', 'Category'])
        fig.update_layout(xaxis_title=None, yaxis_title='Papers (unique IDs)')
    fig.update_traces(marker_color=color, textposition='outside', cliponaxis=False,
                      hovertemplate='%{customdata[1]}<br>%{x} papers<extra></extra>' if horizontal else None)
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
        st.info('No linked records under the current filters.'); return
    table = frame.drop_duplicates(['id', row, col]).groupby([row, col]).id.nunique().unstack(fill_value=0)
    if row_order is None:
        row_order = table.sum(axis=1).sort_values(ascending=False).index.tolist()
    if col_order is None:
        col_order = table.sum().sort_values(ascending=False).index.tolist()
    table = table.reindex(index=row_order, columns=col_order, fill_value=0)
    z = table.astype(float)
    if percent:
        z = z.div(z.sum(axis=0).replace(0, 1), axis=1) * 100
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(tr(c)) for c in table.columns], y=[tr(i) for i in table.index],
        customdata=table.values, colorscale=SEQ,
        colorbar=dict(title='%' if percent else 'Papers'), text=z.round(1).values if percent else table.values,
        texttemplate='%{text:.1f}%' if percent else '%{text}',
        hovertemplate='%{y} × %{x}<br>%{customdata} papers<extra></extra>'))
    fig.update_layout(xaxis_title=None, yaxis_title=None, yaxis_autorange='reversed')
    chart(fig, key, height=height)


def paper_evidence(frame, key, label='Papers and evidence', expanded=False):
    with st.expander(f'{label} · {len(frame):,} papers', expanded=expanded):
        if frame.empty:
            st.info('No matching papers.'); return
        view = frame[['id', 'year', 'title', 'type_name', 'tasks', 'theme', 'model_variants', 'datasets']].copy()
        view['title'] = view.title.map(tr)
        for c in ('type_name', 'theme'):
            view[c] = view[c].map(tr)
        for c in ('tasks', 'model_variants', 'datasets'):
            view[c] = view[c].apply(lambda x: '; '.join(map(str, tr_list(list(x)))))
        view.columns = ['Paper ID', 'Year', 'Title', 'Type', 'Vision task', 'Theme', 'Models', 'Training data source']
        view = view.sort_values(['Year', 'Paper ID'], ascending=[False, True])
        event = st.dataframe(view, hide_index=True, width='stretch', height=320,
                            on_select='rerun', selection_mode='single-row', key=f'{key}_table',
                            column_config={'Title': st.column_config.TextColumn(width='large')})
        st.download_button('Download this paper list · CSV', view.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'{key}_papers.csv', mime='text/csv', key=f'{key}_download')
        selected_rows = event.selection.rows
        if selected_rows:
            rid = view.iloc[selected_rows[0]]['Paper ID']
        else:
            rid = st.selectbox('Choose a paper to see its evidence', view['Paper ID'].tolist(),
                               format_func=lambda v: v + ' · ' + tr(frame.set_index('id').loc[v, 'title'])[:75], key=f'{key}_paper')
        r = frame.set_index('id').loc[rid]
        st.markdown(f'<div class="evidence"><div class="id">{html.escape(rid)} · {html.escape(tr(r["type_name"]))}</div>'
                    f'<h4>{html.escape(tr(r["title"]))}</h4>{html.escape(tr(r["evidence"]) or "No evidence excerpt from the current source.")}</div>', unsafe_allow_html=True)
        if r.get('reclassified'):
            st.caption('Originally an algorithm / comparison study, but no model was trained (ready-made or zero-shot models only), so it is counted as an application study (class = whether a model was trained).')
        a, b = st.columns(2)
        with a:
            st.caption('Objects recognised: ' + (tr(r['classes']) or 'not extracted separately'))
            st.caption('Indicators: ' + (tr(r['indicators']) or 'not extracted separately'))
        with b:
            st.caption('Imagery source: ' + (tr(r['imagery']) or 'not stated'))
            st.caption('Framework / model channel: ' + (tr(r['framework']) or 'not stated'))
        if r['doi']:
            st.link_button('Open paper DOI', 'https://doi.org/' + r['doi'].removeprefix('https://doi.org/'))
        st.write('**Model record status**: ' + tr(r['model_status']))
        if r['theme_detail']:
            st.write('**Theme detail**: ' + tr(r['theme_detail']))
        if r['comparison_models']:
            st.write('**Comparison baselines (listed separately)**: ' + '; '.join(map(str, tr_list(list(r['comparison_models'])))))
        st.write('**Training data status**: ' + tr(r['dataset_status']))
        if r['dataset_records']:
            dr = pd.DataFrame(r['dataset_records']).replace({'role': {'pretrain': '预训练', 'train': '训练／微调', 'unspecified': '训练来源（角色未细分）', 'released': '新发布数据集'}})
            st.dataframe(tr_df(dr).rename(columns={'name': 'Dataset', 'role': 'Data role', 'model': 'Model', 'model_role': 'Model role'}), hide_index=True, width='stretch')
        if r['enrichment_notes'] or r['enrichment_sources']:
            st.markdown('**Additions and checks**')
            for note in r['enrichment_notes']:
                st.write('• ' + tr(note))
            st.dataframe(tr_df(pd.DataFrame(r['enrichment_sources'])).rename(columns={
                'field': 'Field', 'source': 'Evidence source', 'locator': 'Location', 'evidence': 'Evidence / excerpt'}),
                hide_index=True, width='stretch')


# ---------------------------------------------------------------- pages
def landscape():
    hero('01 / RESEARCH LANDSCAPE', 'Research landscape', 'Papers are split by whether the authors trained a model: application studies use ready-made models directly; algorithm & method studies train, fine-tune or build their own annotations.')
    app = papers[papers.group == '应用类']; meth = papers[papers.group == '算法与方法类']
    metrics([('Included', f'{N_ALL:,}', 'Official project list, unique paper IDs.'),
             ('Application', f'{len(app):,}', 'Ready-made or zero-shot models used directly, incl. algorithm/comparison studies that trained no model.'),
             ('Algorithm & method', f'{len(meth):,}', 'Model training, fine-tuning or self-built annotation.')])
    prisma = catalog['prisma']
    st.caption(f'PRISMA: {prisma["assessed"]:,} full texts assessed, {prisma["excluded"]} excluded at full text, {prisma["included"]} included. All {catalog["scope_sync"]["verified_files"]} PDFs in the inclusion index were verified to exist.')
    st.write('')
    a, b = st.columns(2, gap='large')
    with a:
        st.markdown(f'<div class="def"><span class="tag" style="color:{TEAL}">APPLICATION · {len(app)} PAPERS</span><br>'
                    'The authors <b>do not train a model</b>. They use ready-made pretrained or zero-shot models (DeepLabv3+, PSPNet, SegFormer, SAM, etc.) to extract street-scene elements '
                    f'for urban research, or compare off-the-shelf models. {int(app.reclassified.sum())} of these were originally algorithm/comparison studies.'
                    '<br>Answers: <b>which models can be used directly?</b></div>', unsafe_allow_html=True)
    with b:
        sub = ', '.join(f'{tr(k).lower()} {v}' for k, v in meth.type_name.value_counts().items())
        st.markdown(f'<div class="def"><span class="tag" style="color:{NAVY}">ALGORITHM & METHOD · {len(meth)} PAPERS</span><br>'
                    f'The authors <b>train, fine-tune or modify models</b>, or build their own annotated street-view datasets. Sub-types: {sub}.'
                    '<br>Answers: <b>which models transfer well, and how many labels are needed?</b></div>', unsafe_allow_html=True)
    st.write('')
    left, right = st.columns([1, 1.4], gap='large')
    with left:
        st.subheader('Share of the two classes')
        fig = go.Figure(go.Pie(labels=[tr(g) for g in GROUPS], values=[len(app), len(meth)], hole=.68, marker=dict(colors=[BLUE, ORANGE], line=dict(color=SURFACE, width=2)),
                               textinfo='label+percent', textposition='outside', sort=False))
        fig.update_layout(showlegend=False, annotations=[dict(text=f'<b>{N_ALL}</b><br>papers', x=.5, y=.5, showarrow=False, font_size=22)])
        chart(fig, 'classification_pie', height=380)
    with right:
        st.subheader('Algorithm & method sub-types')
        bar(meth.type_name.value_counts(), 'method_types', color=ORANGE, height=380)
    st.subheader('Papers per year, by class')
    trend = papers.groupby(['year', 'group']).id.nunique().reset_index(name='Papers')
    trend['group'] = trend.group.map(tr)
    fig = px.bar(trend, x='year', y='Papers', color='group', barmode='stack', color_discrete_map=GROUP_COLOR_EN,
                 labels={'year': 'Publication year', 'group': 'Class'}, category_orders={'group': [tr(g) for g in GROUPS]})
    fig.update_xaxes(dtick=1)
    chart(fig, 'all_years', height=340)
    cutoff = int(catalog['review_meta']['updated_at'][:4])
    future = int((papers.year > cutoff).sum())
    st.caption(f'Counted by publication year; {cutoff} is the search cut-off year, so its growth cannot be compared directly with complete years.'
               + (f' Another {future} papers carry a year later than the search year; the source-list year is kept.' if future else ''))
    st.caption('Below: "Datasets & object coverage", shared by both classes. For each class\'s detailed analysis, choose "Application" or "Algorithm & method" on the left.')
    if MISSING:
        st.caption(f'Note: {len(MISSING)} papers in the official list are not in the app snapshot and are not shown: ' + ', '.join(MISSING[:20]))


TYPE_DEF = [
    ('迁移与适配', 'Take a ready-made model and train or fine-tune it on street-view data the authors annotated themselves; the architecture is largely unchanged.',
     'e.g. train YOLOv8 on 1,000 self-annotated street-view images to detect potholes and utility poles.'),
    ('算法开发', 'Propose a new model or improve an existing one (architecture, loss, training strategy, etc.), then compare it with other methods.',
     'e.g. add an attention module to DeepLabv3+ to improve street-view segmentation accuracy.'),
    ('数据集研究', 'The main contribution is a new annotated street-view detection/segmentation dataset, usually with baseline model results.',
     'e.g. public datasets of street furniture, building facades or street vendors.'),
    ('模型比较', 'The authors train several existing models themselves and compare them on the same street-view data.',
     'e.g. YOLOv5, Faster R-CNN and SSD compared on accuracy and speed on the same annotations.'),
]


def method_intro():
    allm = papers[papers.group == '算法与方法类']
    cols = st.columns(4, gap='medium')
    for col, (name, what, eg) in zip(cols, TYPE_DEF):
        n = int((allm.type_name == name).sum())
        col.markdown(f'<div class="def"><span class="tag" style="color:{TYPE_COLOR[name]}">{tr(name).upper()} · {n} PAPERS</span><br>'
                     f'{what}<br><span style="color:#8a8984;font-size:13px">{eg}</span></div>', unsafe_allow_html=True)
    st.write('')
    a, b = st.columns([3, 1], gap='medium')
    a.markdown('<div class="def"><span class="tag" style="color:#eb6834">WHAT THE "TRAINING & TRANSFER" PAGE IS</span><br>'
               'It answers the two wave-2 questions: <b>(1) when no ready-made model fits, which base model transfers well to street view; '
               '(2) how many images need to be labelled, and how large a dataset is enough.</b> The data come from a dedicated full-text extraction of all 491 papers: '
               'which base model was used, which weights training started from, how many images were self-annotated, and how much scores improved after training.</div>', unsafe_allow_html=True)
    b.button('Open Training & transfer →', type='primary', on_click=lambda: st.session_state.update(page='训练与迁移'), key='open_training', width='stretch')
    st.write('')


def overview():
    color = GROUP_COLOR[group]
    if group == '应用类':
        hero('03 / APPLICATION STUDIES', 'What do application studies focus on?', 'Publication years, vision tasks and research themes show how ready-made models are used in urban research.', group)
    else:
        hero('03 / METHOD STUDIES', 'What do algorithm & method studies do?', 'What these papers share: the authors trained, fine-tuned or modified a model, or annotated street-view data themselves. They are split into 4 sub-types by main contribution.', group)
        method_intro()
    if selected.empty:
        st.info('No papers match the current filters; widen the filters on the left.'); return
    known = selected.qualified_models.apply(bool).sum()
    if group == '应用类':
        metrics([('In scope', f'{len(selected):,}', f'{N_SCOPE} application papers in total.'),
                 ('Main model named', f'{known:,}', 'Papers with a complete model or model-family description.'),
                 ('Theme coded', f'{(selected.theme != "主题未编码").sum():,}', 'Paper-by-paper coding of research question, outcome and technical theme.'),
                 ('Training source recorded', f'{selected.datasets.apply(bool).sum():,}', "The model's pretraining/training dataset is recorded.")])
    else:
        tc = selected.type_name.value_counts()
        metrics([('In scope', f'{len(selected):,}', f'{N_SCOPE} algorithm & method papers in total.'),
                 ('Transfer & adaptation', f'{tc.get("迁移与适配", 0):,}', 'An existing model trained or fine-tuned on self-annotated street-view data.'),
                 ('Algorithm development', f'{tc.get("算法开发", 0):,}', 'A new or improved model.'),
                 ('Dataset + comparison', f'{tc.get("数据集研究", 0) + tc.get("模型比较", 0):,}', 'A street-view dataset release, or several models trained and compared.')])
    st.write('')
    left, right = st.columns([1.25, 1], gap='large')
    with left:
        st.subheader('Papers per year')
        if group == '应用类':
            annual = selected.groupby('year').id.nunique().reindex(range(years[0], years[1] + 1), fill_value=0)
            fig = px.bar(x=annual.index, y=annual.values, labels={'x': 'Publication year', 'y': 'Papers'}, text=annual.values)
            fig.update_traces(marker_color=color, textposition='outside')
        else:
            annual = selected.groupby(['year', 'type_name']).id.nunique().reset_index(name='Papers')
            annual['type_name'] = annual.type_name.map(tr)
            fig = px.bar(annual, x='year', y='Papers', color='type_name', barmode='stack',
                         labels={'year': 'Publication year', 'type_name': 'Sub-type'}, color_discrete_map=TYPE_COLOR_EN)
        fig.update_xaxes(dtick=1)
        chart(fig, 'scope_years', height=380)
    with right:
        st.subheader('Vision tasks')
        event = bar(counts_for(selected, 'tasks'), 'task_counts', color=color, selectable=True, height=380)
    st.caption('A paper can have several tasks (e.g. both detection and segmentation), so task counts can add up to more than the number of papers. Click a bar to list its papers.')
    task_selection = chosen_point(event)
    if task_selection:
        paper_evidence(selected[selected.tasks.apply(lambda ts: task_selection in ts)], 'task_drill', f'Selected in chart: {tr(task_selection)}', True)
    if group == '算法与方法类':
        st.subheader('Sub-type × vision task')
        st_rows = selected[['id', 'type_name', 'tasks']].explode('tasks')
        matrix(st_rows, 'type_name', 'tasks', 'type_task', height=300, color=NAVY,
               row_order=[t for t in ['迁移与适配', '算法开发', '数据集研究', '模型比较'] if t in set(st_rows.type_name)])
    st.subheader('Research themes over time' if group == '应用类' else 'Technical themes over time')
    left, right = st.columns([1, 1.55], gap='large')
    with left:
        event_theme = bar(selected.theme.value_counts(), 'theme_counts', color=color, selectable=True, height=500)
    with right:
        mode = st.radio('Theme-by-year chart', ['论文数', '年内占比'], horizontal=True, key='theme_mode',
                        format_func={'论文数': 'Papers', '年内占比': 'Share within year'}.get)
        matrix(selected[['id', 'year', 'theme']], 'theme', 'year', 'theme_years', height=455, color=color,
               percent=mode == '年内占比', col_order=list(range(years[0], years[1] + 1)))
    theme_selection = chosen_point(event_theme)
    if theme_selection:
        paper_evidence(selected[selected.theme == theme_selection], 'theme_drill', f'Selected in chart: {tr(theme_selection)}', True)
    paper_evidence(selected, 'overview')


def models_page():
    color = GROUP_COLOR[group]
    hero('04 / MODELS & DATA', 'Where do the models come from, and what are they used for?',
         'Model family, specific architecture, training source and usage role are recorded separately. ' + ('Application studies: "ready-made model + pretraining dataset".' if group == '应用类' else 'Algorithm & method studies: "base model + initialisation/training data".'), group)
    if selected.empty:
        st.info('No papers match the current filters.'); return
    role_choice = st.radio('Model role', ['主用分割／检测模型', '主用＋实际运行的比较分支'], horizontal=True, key='model_role_scope',
                           format_func={'主用分割／检测模型': 'Main detection/segmentation model', '主用＋实际运行的比较分支': 'Main + comparison models actually run'}.get)
    roles = ['main'] if role_choice.startswith('主用分割') else ['main', 'comparison']
    use_rows = all_models[all_models.id.isin(ids) & all_models.role.isin(roles)].copy()
    named_rows = use_rows[use_rows.kind == QUALIFIED]
    use_pairs = all_pairs[all_pairs.id.isin(ids) & all_pairs.model_role.isin(roles)].copy()
    st.caption('Main = the model the paper uses to produce its results (for algorithm & method studies, the base model the authors trained/modified); comparison = other models the authors also ran for comparison (models only mentioned in the literature review do not count).')
    known_keys = set(zip(named_rows.id, named_rows.model))
    relevant = use_pairs.loc[[(rid, m) in known_keys for rid, m in zip(use_pairs.id, use_pairs.model)]].copy()
    metrics([('Papers', len(selected), 'Current filter scope.'),
             ('Model/family named', named_rows.id.nunique(), 'Unique paper IDs.'),
             ('Training source recorded', selected.datasets.apply(bool).sum(), 'Public training sets and self-built training sets.'),
             ('Model–dataset pairs', relevant.id.nunique(), 'A model and dataset are linked only with evidence; no Cartesian pairing.')])
    left, right = st.columns([1.3, 1], gap='large')
    with left:
        st.subheader('Most used models')
        c1, c2 = st.columns(2)
        top_n = c1.slider('Show top N models', 5, 30, 15, key='model_top')
        name_scope = c2.radio('Name level', ['模型家族', '模型 + 骨干／规模'], horizontal=True, key='name_scope',
                              format_func={'模型家族': 'Model family', '模型 + 骨干／规模': 'Model + backbone/size'}.get)
        model_col = 'model' if name_scope == '模型家族' else 'variant'
        mc = named_rows.groupby(model_col).id.nunique().sort_values(ascending=False)
        event = bar(mc.head(top_n), 'models_bar', color=color, selectable=True, height=520)
    with right:
        st.subheader('Pretraining datasets' if group == '应用类' else 'Training / pretraining data sources')
        training_records = pd.DataFrame(catalog['dataset_records'])
        allowed_roles = ['main', 'paper'] if roles == ['main'] else ['main', 'comparison', 'paper']
        training_records = training_records[training_records.id.isin(ids) & training_records.model_role.isin(allowed_roles)]
        training_records = training_records[training_records.role.isin(['pretrain', 'train', 'unspecified'])]
        dc = training_records.groupby('name').id.nunique().sort_values(ascending=False)
        bar(dc.head(12), 'dataset_freq', color=INK2, height=600)
    picked = chosen_point(event)
    if picked:
        paper_evidence(selected[selected.id.isin(named_rows[named_rows[model_col] == picked].id)], 'model_drill', f'Selected in chart: {tr(picked)}', True)
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('Model × data source')
        top_models = relevant.groupby('model').id.nunique().sort_values(ascending=False).head(top_n).index.tolist()
        top_data = relevant.groupby('dataset').id.nunique().sort_values(ascending=False).head(8).index.tolist()
        matrix(relevant[relevant.model.isin(top_models) & relevant.dataset.isin(top_data)], 'model', 'dataset', 'model_data_heatmap',
               height=max(380, len(top_models) * 28 + 120), row_order=top_models, col_order=top_data, color=color)
    with right:
        st.subheader('Model × vision task')
        mt = named_rows[['id', 'model', 'tasks']].explode('tasks').dropna(subset=['tasks'])
        best = mt.groupby('model').id.nunique().sort_values(ascending=False).head(top_n).index.tolist()
        matrix(mt[mt.model.isin(best)], 'model', 'tasks', 'model_task_heatmap', height=max(380, len(best) * 28 + 120), row_order=best, color=color)
    st.caption('Each cell = unique papers. Benchmark results, validation sets, street-view imagery platforms and datasets of auxiliary models are not counted as the model\'s training source.')
    st.subheader('Training data source × ' + ('research theme' if group == '应用类' else 'technical theme'))
    ds_papers = selected[['id', 'datasets', 'theme']].explode('datasets').dropna(subset=['datasets'])
    top_ds = ds_papers.groupby('datasets').id.nunique().sort_values(ascending=False).head(10).index.tolist()
    matrix(ds_papers[ds_papers.datasets.isin(top_ds)], 'theme', 'datasets', 'dataset_themes', height=520, col_order=top_ds)
    st.caption('Training/pretraining datasets used by the papers × theme, unique IDs, follows the filters on the left. Dataset specifications and classes are in the lower half of the "Landscape & datasets" page.')
    paper_evidence(selected, 'models')


def dataset_page():
    st.divider()
    hero('02 / DATASETS & OBJECTS', 'Datasets & object coverage', 'Shared by both classes: which tasks the common public urban datasets support, their size and collection context, which urban objects they can recognise, and which label schemes cover a target element.')
    tabs = st.tabs(['① Supported tasks', '② Dataset comparison', '③ Objects & classes', '④ Find by element'])
    names = dataset_df.name.tolist()
    with tabs[0]:
        st.subheader('Which vision tasks does each dataset support?')
        def native_tasks(d):  # from the 标注类型 (annotation type) column of 数据集表_v2.xlsx; '框' (boxes) = object detection
            if d['name'] == 'SA-1B':
                return ['无类别掩码分割']
            import re
            a = d['annotation'] or ''
            out = ['目标检测'] if re.search(r'检测|2D|(?<!3D )框', a) else []  # 2D boxes = detection; 3D-only boxes are not counted
            out += [label for w, label in [('语义', '语义分割'), ('实例', '实例分割'), ('全景', '全景分割')] if w in a]
            return out
        native = [{'Dataset': d['name'], 'Vision task': tr(t)} for d in catalog['datasets'] for t in native_tasks(d)]
        table = pd.DataFrame(native).assign(value=1).pivot_table(index='Dataset', columns='Vision task', values='value', aggfunc='max', fill_value=0)
        fig = go.Figure(go.Heatmap(z=table.values, x=table.columns, y=table.index,
                         colorscale=[[0, '#f3f6fa'], [1, '#3987e5']], showscale=False,
                         text=table.replace({0: '—', 1: '●'}).values, texttemplate='%{text}',
                         hovertemplate='%{y} · %{x}<br>%{text}<extra></extra>'))
        chart(fig, 'native_tasks', height=440)
        st.caption('Derived from the "annotation type" column of the project dataset table: boxes or detection → object detection; semantic / instance / panoptic → the matching segmentation. Describes only the subset in the table (e.g. only the IDD scene-parsing subset). SA-1B masks have no semantic classes.')
    with tabs[1]:
        compare = st.multiselect('Datasets to compare', names, default=names, key='compare_ds')
        use_app = counts_for(papers[papers.group == '应用类'], 'datasets')
        use_meth = counts_for(papers[papers.group == '算法与方法类'], 'datasets')
        rows = []
        for d in catalog['datasets']:
            if d['name'] in compare:
                rows.append({'Dataset': d['name'], 'Application papers': int(use_app.get(d['name'], 0)), 'Algorithm & method papers': int(use_meth.get(d['name'], 0)),
                             'Supported tasks': '; '.join(tr_list(native_tasks(d))), 'Annotation (source table)': tr(d['annotation']), 'Scene': tr(d['scene']),
                             'Classes / scheme': tr(d['classes']), 'Size & splits': tr(d['size']), 'Collection context': tr(d['geography']), 'Fit to street view': tr(d['fit'])})
        view = pd.DataFrame(rows)
        st.dataframe(view, hide_index=True, width='stretch', height=360)
        st.caption('Paper counts = papers of that class using the dataset as a training/pretraining source (unique IDs).')
        st.download_button('Download dataset comparison · CSV', view.to_csv(index=False).encode('utf-8-sig'), 'dataset_comparison.csv', 'text/csv')
        if compare:
            detail = st.selectbox('Dataset details', compare, key='dataset_detail')
            d = next(d for d in catalog['datasets'] if d['name'] == detail)
            st.markdown(f'**{tr(d["full_name"])}** · {d["year"]}')
            for label, field in [('Reference', 'reference'), ('Imagery', 'imagery'), ('Resolution', 'resolution'), ('Licence', 'license')]:
                st.write(f'**{label}**: {tr(d[field])}')
            st.caption(f'Source: project dataset master table, sheet 1 (dataset overview), row {d["source_row"]}.')
    with tabs[2]:
        st.subheader('Urban-object coverage matrix')
        el_en = dict(zip(coverage_df.element, coverage_df.english))
        variants = coverage_df.variant.drop_duplicates().tolist()
        chosen_variants = st.multiselect('Label schemes / versions to compare', variants,
             default=variants, key='coverage_variants')
        elements = coverage_df.element.drop_duplicates().tolist()
        chosen_elements = st.multiselect('Only some objects (empty = all)', elements, key='coverage_elements', format_func=el_en.get)
        cov = coverage_df[coverage_df.variant.isin(chosen_variants)]
        if chosen_elements:
            cov = cov[cov.element.isin(chosen_elements)]
        if not cov.empty:
            c = cov.pivot(index='element', columns='variant', values='coverage').reindex(index=[e for e in elements if e in set(cov.element)], columns=chosen_variants)
            labels = cov.pivot(index='element', columns='variant', values='labels').reindex(index=c.index, columns=c.columns).map(tr)
            fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=[el_en[e] for e in c.index], zmin=0, zmax=2,
                     customdata=labels.values, showscale=False,
                     colorscale=[[0, '#f0efec'], [.249, '#f0efec'], [.25, '#b7d3f6'], [.749, '#b7d3f6'], [.75, '#3987e5'], [1, '#3987e5']],
                     text=c.replace({0: '—', 1: '≈', 2: '✓'}).values, texttemplate='%{text}',
                     hovertemplate='%{y}<br>%{x}<br>%{customdata}<extra></extra>'))
            fig.update_layout(yaxis_autorange='reversed')
            chart(fig, 'coverage', height=max(400, len(c) * 26 + 130))
            st.download_button('Download this coverage matrix · CSV', labels.rename(index=el_en).to_csv().encode('utf-8-sig'), 'object_coverage.csv', 'text/csv')
        st.caption('✓ own class　≈ only inside a broader class　— no matching class. Class coverage says nothing about a model\'s actual recognition accuracy.')
        st.subheader('Full class lists')
        ds = st.selectbox('Which dataset\'s object labels?', names, key='classlist_ds')
        for r in catalog['classlists']:
            if r['dataset'] == ds:
                with st.expander(f'{r["variant"]} · {tr(r["subset"])} · {r["count"]} classes', expanded=True):
                    st.write(tr(r['labels']))
    with tabs[3]:
        st.subheader('Find label schemes for target objects')
        el_en = dict(zip(coverage_df.element, coverage_df.english))
        wanted = st.multiselect('Which urban elements do you want to recognise?', coverage_df.element.drop_duplicates().tolist(),
                               default=coverage_df.element.drop_duplicates().tolist(), key='wanted_objects', format_func=el_en.get)
        strict = st.toggle('Keep only schemes with an own class for every target', value=False, key='strict_coverage')
        if wanted:
            cov = coverage_df[coverage_df.element.isin(wanted)]
            summary = cov.groupby('variant').coverage.agg(['min', 'sum'])
            valid = summary.index[summary['min'] == 2] if strict else summary.sort_values('sum', ascending=False).index
            result = cov[cov.variant.isin(valid)].pivot(index='variant', columns='element', values='labels').reindex(columns=wanted)
            if result.empty:
                st.info('No scheme has an own class for all of these. Turn off the strict option to see broader classes.')
            else:
                st.dataframe(result.map(tr).rename(columns=el_en), width='stretch')
            st.caption('Matched on the class tables only; available weights, performance in the target region and annotation cost need checking separately.')


BINS = [(1, 100, '≤100'), (101, 300, '101–300'), (301, 1000, '301–1k'), (1001, 3000, '1k–3k'), (3001, 10000, '3k–10k'), (10001, 10**9, '>10k')]


def training_page():
    hero('05 / TRAINING & TRANSFER', 'Training & transfer: how many labels, and which models transfer well?',
         'The two wave-2 questions: (1) how many labels / how large a dataset is enough; (2) when no ready-made model fits, which base model transfers well. Data from a dedicated full-text extraction of the algorithm & method studies.', group)
    st.markdown('<div class="def">"Transfer" means taking a model trained on a large dataset (COCO, Cityscapes, ImageNet, etc.) and continuing to train or fine-tune it for a street-view task. '
                'This page is ordered by strength of evidence: first <b>direct comparisons within the same paper</b> (most reliable), then <b>actual labelling scale and starting weights</b> (describes current practice), '
                'and finally <b>which models are common and often win</b> (indicative only).</div>', unsafe_allow_html=True)
    st.write('')
    flat_all, gains_all, h2h_all = read_wave2(file_version(WAVE2 / 'wave2_flat.csv', WAVE2 / 'gains_curated.py', WAVE2 / 'wave2_h2h.csv'))
    curves_all, ntypes = read_curves(file_version(WAVE2 / 'learning_curves.json', WAVE2 / 'note_types.csv'))
    flat = flat_all[flat_all.id.isin(ids)].copy()
    gains = gains_all[gains_all.id.isin(ids)].copy()
    if flat.empty:
        st.info('No extraction records under the current filters.'); return
    own = flat[flat.data_source.isin(['own-annotated', 'both'])].copy()
    for c_ in ('own_images', 'own_train', 'own_val', 'own_test'):
        own[c_] = pd.to_numeric(own[c_], errors='coerce')
    counted = own.dropna(subset=['own_images']).copy()
    # own labels used only for evaluation (zero-shot papers, or only a test split was annotated) are not training labels
    eval_only = (counted.training_mode == 'zero-shot (no training)') | (counted.own_train.isna() & counted.own_val.isna() & counted.own_test.notna() & (counted.own_test == counted.own_images))
    n_eval_only = int(eval_only.sum()); counted = counted[~eval_only].copy()
    g = gains.copy()
    g['标注图像'] = pd.to_numeric(g['适配用标注图像'], errors='coerce')
    g['提升'] = pd.to_numeric(g['提升(点)'], errors='coerce')
    ga = g[(g['类型'] == 'A') & g['标注图像'].gt(0) & g['提升'].notna()]
    curves = [c for c in curves_all if c['id'] in ids]

    # ---- conclusion cards
    st.markdown('### Key findings on this page')
    c = st.columns(4, gap='medium')
    card = lambda col, tag, body: col.markdown(f'<div class="def"><span class="tag" style="color:{ORANGE}">{tag}</span><br>{body}</div>', unsafe_allow_html=True)
    if len(ga):
        card(c[0], 'A LITTLE LOCAL LABELLING HELPS', f'{len(ga)} within-paper pairs: fine-tuning on images from the target city used a median of <b>{ga["标注图像"].median():,.0f} images</b> and gained a median of <b>{ga["提升"].median():.1f} percentage points</b> ({ga["提升"].min():.0f}–{ga["提升"].max():.0f}).')
    curve_ids = sorted({x['id'] for x in curves if x['type'] == 'curve'})
    flat_pl = [x for x in curves if x['type'] == 'curve' and any(k in (x.get('plateau_note') or '') for k in ('不再', '无提升', '无明显提升'))]
    flat_pl.sort(key=lambda x: (0 if '张' in x['plateau_note'] else 1, x['id']))
    rising = [x for x in curves if x['type'] == 'curve' and any(k in (x.get('plateau_note') or '') for k in ('仍在上升', '未见平台'))]
    body = f'{len(curve_ids)} papers trained with different label amounts within the same paper. '
    if flat_pl:
        body += 'Plateaus: ' + '; '.join(f'{x["id"]} {tr(x["plateau_note"])}' for x in flat_pl[:3]) + '. '
    if rising:
        body += 'Still rising at the full set: ' + ', '.join(x['id'] for x in rising[:3]) + '.'
    card(c[1], 'DIMINISHING RETURNS (WITHIN-PAPER)', body)
    if len(counted):
        q = counted.own_images.quantile([.25, .5, .75])
        med = counted.groupby('task_group').own_images.median()
        card(c[2], 'ACTUAL LABELLING SCALE', f'Median self-annotated images <b>{q[.5]:,.0f}</b> (middle half {q[.25]:,.0f}–{q[.75]:,.0f}); detection {med.get("检测", float("nan")):,.0f}, segmentation {med.get("分割", float("nan")):,.0f}. This is "how many were labelled", not "the minimum needed"; another {n_eval_only} papers used their own labels for testing only and are not counted.')
    ns = (flat.init == 'not stated').mean() * 100
    card(c[3], 'STARTING WEIGHTS', f'Detection mostly starts from <b>COCO</b>, segmentation from <b>ImageNet backbones / Cityscapes</b>; <b>{ns:.0f}%</b> of papers do not state the initial weights.')
    st.write('')

    # ---- 1. before / after
    st.markdown('### ① Same model, same test set: how much does fine-tuning on local labels help?')
    if gains.empty:
        st.info('No before/after comparisons under the current filters.')
    else:
        left, right = st.columns([1, 1.25], gap='large')
        with left:
            if ga.empty:
                st.info('No numeric records to plot.')
            else:
                gd = ga.rename(columns={'模型': 'Model', '指标': 'Metric', '适配前': 'Before', '适配后': 'After', '测试集': 'Test set'})
                for c_ in ('Model', 'Metric', 'Before', 'After', 'Test set'):
                    gd[c_] = gd[c_].map(lambda v: tr(str(v)))
                fig = px.scatter(gd, x='标注图像', y='提升', text='id', log_x=True, hover_data=['Model', 'Metric', 'Before', 'After', 'Test set'],
                                 labels={'标注图像': 'Target-city images used for fine-tuning (log axis)', '提升': 'Gain (percentage points)'})
                fig.update_traces(marker=dict(size=11, color=ORANGE, line=dict(color=SURFACE, width=2)), textposition='top center', textfont=dict(size=10, color=INK2))
                fig.add_hline(y=ga['提升'].median(), line_dash='dot', line_color=MUTED, annotation_text=f'median gain {ga["提升"].median():.1f}', annotation_font_color=INK2)
                chart(fig, 'gain_scatter', height=430)
                st.caption('Each point is one paper: the difference for the same model on the same test set before and after fine-tuning with self-annotated target-city images. Tens to hundreds of images already give a clear gain.')
        with right:
            show = gains[['id', '类型', '模型', '指标', '适配前', '适配后', '提升(点)', '适配用标注图像']].copy()
            show['类型'] = show['类型'].map({'A': '本地自标微调', 'B': '无目标标签', 'C': '预训练/策略/清洗'})
            show = tr_df(show.astype(str))
            show.columns = ['Paper ID', 'Type', 'Model', 'Metric', 'Before', 'After', 'Gain (pts)', 'Labelled images used']
            st.dataframe(show, hide_index=True, width='stretch', height=430)
            st.caption('A = fine-tuned with self-annotated target-city images; B = no target labels (unsupervised adaptation, synthetic data, etc.); C = different pretraining source, training strategy or data cleaning. All checked manually, one by one.')

    # ---- 2. learning curves
    st.markdown('### ② Within-paper learning curves: does the gain stop as more labels are added?')
    if not curves:
        st.info('Learning-curve data not generated or none under the current filters.')
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
                    rows.append(dict(id=x['id'], Curve=series, kind=x['type'], Model=x.get('model', ''), Metric=x.get('metric', ''), unit=x.get('x_unit', ''),
                                     x=float(pt['x']), y=float(pt['y']) * scale, Label=tr(pt.get('label', '')), Plateau=tr(x.get('plateau_note', ''))))
                except (TypeError, ValueError, KeyError):
                    pass
        lc = pd.DataFrame(rows)
        kind = st.radio('Curve type', ['学习曲线（同篇不同标注量）', '加入本地数据前后'], horizontal=True, key='curve_kind',
                        format_func={'学习曲线（同篇不同标注量）': 'Learning curve (different label amounts, same paper)', '加入本地数据前后': 'Before / after adding local data'}.get)
        sub_all = lc[lc['kind'] == ('curve' if kind.startswith('学习') else 'local_add')]
        UNIT = {'images': 'images', 'percent': 'share of training data %', 'instances': 'annotated instances', 'patches': 'patches', 'other': 'other (configuration order)'}

        def curve_chart(sub, unit, key):
            sub = sub[sub['unit'] == unit].sort_values(['Curve', 'x'])
            if sub.empty:
                st.info('No records of this type.'); return
            logx = unit in ('images', 'instances', 'patches')
            sub = sub.assign(x=sub.x.where(~(logx & sub.x.eq(0)), 1))  # 0 (no local data) is drawn at 1 on the log axis
            fig = px.line(sub, x='x', y='y', color='Curve', markers=True, log_x=logx,
                          hover_data=['Model', 'Metric', 'Label', 'Plateau'], color_discrete_sequence=COLORS,
                          labels={'x': {'images': 'Annotated images', 'other': 'Configuration order'}.get(unit, UNIT[unit].capitalize()), 'y': 'Score (rescaled to 0–100)', 'Curve': 'Paper'})
            fig.update_traces(line=dict(width=2), marker=dict(size=8))
            chart(fig, key, height=460)

        st.markdown(f'**x-axis: annotated images** · {sub_all[sub_all["unit"] == "images"]["Curve"].nunique()} curves')
        curve_chart(sub_all, 'images', 'curves')
        st.caption('Each line is one paper (different cities/datasets in the same paper are drawn separately). Metrics differ, so read only each line\'s own trend (does it flatten?) and do not compare heights across lines. Metrics on a 0–1 scale are multiplied by 100; the x-axis is logarithmic and "0 images (no local data)" is drawn at 1.')
        others = [u for u in ['instances', 'percent', 'patches', 'other'] if (sub_all['unit'] == u).any()]
        if others:
            with st.expander('Curves with other x-axis units (' + ', '.join(f'{UNIT[u]}: {sub_all[sub_all["unit"] == u]["Curve"].nunique()}' for u in others) + ')'):
                u = st.radio('x-axis unit', others, horizontal=True, key='curve_unit', format_func=lambda x: UNIT[x])
                curve_chart(sub_all, u, 'curves_other')
                st.caption('These papers did not report image counts: instances = number of annotated objects, share = percentage of the training set, patches = tiles cut from large images, "other" = configuration order only. Read the trend only; it does not map onto the image counts above.')
    with st.expander('What each paper reports on "label amount vs performance" (by type)', expanded=False):
        notes = flat[flat.label_efficiency_note.fillna('').str.len() > 30][['id', 'base_model', 'own_images', 'label_efficiency_note']].copy()
        if ntypes is not None:
            notes = notes.merge(ntypes, on='id', how='left')
            notes['type'] = notes['type'].map({'curve': '学习曲线', 'local_add': '加入本地数据', 'imbalance': '类别不平衡', 'qualitative': '定性说法'}).fillna('未分类')
            order = {'学习曲线': 0, '加入本地数据': 1, '类别不平衡': 2, '定性说法': 3, '未分类': 4}
            notes = notes.sort_values('type', key=lambda s: s.map(order))
            pick = st.multiselect('Types shown', list(order), default=['学习曲线', '加入本地数据'], key='note_types', format_func=tr)
            notes = notes[notes['type'].isin(pick)]
            view = tr_df(notes[['id', 'type', 'summary_zh', 'base_model', 'own_images', 'label_efficiency_note']])
            view.columns = ['Paper ID', 'Type', 'Key point', 'Base model', 'Self-annotated images', 'Original note']
        else:
            view = notes.rename(columns={'id': 'Paper ID', 'base_model': 'Base model', 'own_images': 'Self-annotated images', 'label_efficiency_note': 'Original note'})
        st.dataframe(view.astype(str), hide_index=True, width='stretch', height=420,
                     column_config={'Original note': st.column_config.TextColumn(width='large')})
        st.caption('"Learning curve" and "adding local data" give paired numbers and count as evidence; "class imbalance" signals that samples must be balanced per class; "qualitative statement" is background only.')

    # ---- 3. labelling scale and starting weights
    st.markdown('### ③ Actual labelling scale and starting weights (current practice)')
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('How many street-view images did the authors annotate?')
        counted['bin'] = counted.own_images.apply(lambda n: next(b[2] for b in BINS if b[0] <= n <= b[1]))
        dist = counted.groupby(['bin', 'task_group']).id.nunique().reset_index(name='Papers')
        dist['task_group'] = dist.task_group.map(tr)
        fig = px.bar(dist, x='bin', y='Papers', color='task_group', barmode='stack',
                     category_orders={'bin': [b[2] for b in BINS], 'task_group': [tr(t) for t in ['检测', '分割', '其他']]},
                     color_discrete_map=TASK_COLOR_EN, labels={'task_group': 'Task', 'bin': 'Self-annotated images'})
        chart(fig, 'own_dist', height=380)
        stats = counted.groupby('task_group').own_images.describe()[['count', '25%', '50%', '75%']].round(0)
        stats.columns = ['Papers', 'P25', 'Median', 'P75']; stats.index = [tr(i) for i in stats.index]; stats.index.name = 'Task'
        st.dataframe(stats, width='stretch')
        st.caption(f'Only self-annotated images used for training are counted; another {n_eval_only} papers used their own images for testing only (zero-shot evaluation or only a test set labelled) and are not counted.')
    with right:
        st.subheader('Which weights did the base model start from?')
        fi = flat[['id', 'family', 'init']]
        top_f = fi.groupby('family').id.nunique().sort_values(ascending=False).head(12).index.tolist()
        top_i = fi.groupby('init').id.nunique().sort_values(ascending=False).head(7).index.tolist()
        matrix(fi[fi.family.isin(top_f) & fi.init.isin(top_i)], 'family', 'init', 'family_init', height=520,
               row_order=top_f, col_order=top_i, color=NAVY)
        st.caption('"not stated" means the paper does not say which weights training started from.')

    # ---- 4. which model families
    st.markdown('### ④ Which base models are common and often win (indicative)')
    h2h = h2h_all[h2h_all.id.isin(ids)]
    rows = []
    for fam in sorted({f for fs in h2h.families for f in fs.split('; ') if f}):
        rows.append({'family': fam, 'Compared': int(h2h.families.apply(lambda fs: fam in fs.split('; ')).sum()), 'Rated best': int((h2h.winner == fam).sum())})
    hh = pd.DataFrame(rows)
    min_n = st.slider('Minimum number of comparisons', 2, 15, 5, key='h2h_min')
    hh = hh[hh['Compared'] >= min_n].copy() if not hh.empty else hh
    if hh.empty:
        st.info('Not enough within-paper comparisons under the current filters.')
    else:
        hh['family'] = hh.family.map(tr)
        hh['Win rate'] = (hh['Rated best'] / hh['Compared'] * 100).round(0)
        hh = hh.sort_values(['Win rate', 'Compared'])
        fig = go.Figure()
        fig.add_bar(y=hh['family'], x=hh['Compared'], orientation='h', name='Compared', marker_color='#cde2fb', hovertemplate='%{y}<br>compared %{x} times<extra></extra>')
        fig.add_bar(y=hh['family'], x=hh['Rated best'], orientation='h', name='Rated best', marker_color=ORANGE,
                    text=[f'{w}/{a} · {r:.0f}%' for w, a, r in zip(hh['Rated best'], hh['Compared'], hh['Win rate'])],
                    textposition='outside', textfont=dict(color=INK2), hovertemplate='%{y}<br>rated best %{x} times<extra></extra>')
        fig.update_layout(barmode='overlay', xaxis_title='Papers', yaxis_title=None)
        chart(fig, 'h2h_bar', height=max(380, len(hh) * 32 + 110))
    st.caption('When a paper compares ≥2 model families, each family counts as "compared" once and the one the authors rate best counts as a "win". Improved variants count towards their base family and authors tend to report their own method as best, so win rates are indicative only.')
    with st.expander('Median best score per model family (across papers, weak evidence)'):
        perf = flat.copy()
        for col in ('best_mAP50', 'best_mIoU'):
            perf[col] = pd.to_numeric(perf[col], errors='coerce')
        det = perf[perf.task_group == '检测'].groupby('family').agg(n_mAP50=('best_mAP50', 'count'), mAP50_median=('best_mAP50', 'median'))
        seg = perf[perf.task_group == '分割'].groupby('family').agg(n_mIoU=('best_mIoU', 'count'), mIoU_median=('best_mIoU', 'median'))
        fam = det.join(seg, how='outer').fillna({'n_mAP50': 0, 'n_mIoU': 0})
        fam = fam[(fam['n_mAP50'] + fam['n_mIoU']) > 0]
        fam = fam.assign(total=fam['n_mAP50'] + fam['n_mIoU']).sort_values('total', ascending=False).head(15).round(1)
        fam.index = [tr(i) for i in fam.index]; fam.index.name = 'Model family'
        fam = fam[['n_mAP50', 'mAP50_median', 'n_mIoU', 'mIoU_median']]
        fam.columns = ['Papers with mAP50', 'Median mAP50', 'Papers with mIoU', 'Median mIoU']
        st.dataframe(fam, width='stretch',
                     column_config={'Median mAP50': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f'),
                                    'Median mIoU': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.1f')})
        st.caption('Each paper\'s highest value for the metric, then the median. Classes, test sets and difficulty differ between papers, so this cannot rank models.')
    with st.expander('Across papers: label amount vs score (and why this comparison fails)'):
        metric = st.radio('Metric', ['Detection mAP50', 'Segmentation mIoU'], horizontal=True, key='amount_metric')
        col, tg = ('best_mAP50', '检测') if metric.startswith('Detection') else ('best_mIoU', '分割')
        sc = counted[(counted.task_group == tg)].copy()
        sc[col] = pd.to_numeric(sc[col], errors='coerce'); sc = sc.dropna(subset=[col])
        if not sc.empty:
            topf = sc.family.value_counts().head(6).index
            sc['Model family'] = sc.family.where(sc.family.isin(topf), '其他').map(tr)
            fig = px.scatter(sc, x='own_images', y=col, color='Model family', log_x=True, hover_data=['id', 'base_model', 'n_classes'],
                             labels={'own_images': 'Self-annotated images (log axis)', col: metric.split()[1] + ' (%)'})
            fig.update_traces(marker=dict(size=9, opacity=.85, line=dict(color=SURFACE, width=1.5)))
            chart(fig, 'amount_effect', height=420)
        st.caption('The points form a cloud with no "more labels → higher score" pattern, because classes, test sets and difficulty all differ between papers. Evidence on "how many labels" must come from ① and ②.')
    paper_evidence(selected, 'training')


if page == '研究全景':
    landscape()
    dataset_page()
    st.divider()
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button('Go to application studies →', type='primary', on_click=goto('应用类'), key='enter_app', width='stretch')
    c2.button('Go to algorithm & method studies →', on_click=goto('算法与方法类'), key='enter_method', width='stretch')
elif page == '研究概览':
    overview()
elif page == '模型与数据来源':
    models_page()
else:
    training_page()

st.markdown('<div class="foot">STREETSCAPE REVIEW · unique paper IDs · filters, chart selection, evidence look-up and CSV export</div>', unsafe_allow_html=True)
with st.expander('Data sources and counting rules'):
    st.write('Classes follow the official project list: application = the authors did not train a model and used ready-made or zero-shot models directly; algorithm & method = model training, fine-tuning or self-built annotation.')
    st.write('Model names are standardised: "model family" merges different spellings of the same model (YOLO is split by version); "model + backbone/size" keeps only the backbone or YOLO size and drops case and training-setting differences. The main model of algorithm & method studies comes from the wave-2 full-text extraction.')
    st.write('The "Training & transfer" page comes from the wave-2 full-text extraction (analysis/wave2); before/after comparisons were checked manually one by one.')
    n_app, n_meth = int((papers.group == '应用类').sum()), int((papers.group == '算法与方法类').sum())
    st.write(f'{N_ALL} papers formally included: {n_app} application, {n_meth} algorithm & method. PRISMA: {catalog["prisma"]["assessed"]:,} full texts assessed, {catalog["prisma"]["excluded"]} excluded at full text. All full-text paths were checked against the drive index.')
    st.caption('Study counts follow the official list; unconfirmed training sources are left empty and model–dataset pairs are never guessed.')
    st.caption('In-app evidence snapshot updated: ' + catalog['review_meta']['updated_at'])
