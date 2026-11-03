# Object detection and segmentation on street-view imagery: literature review explorer

Streamlit app for the systematic review: research landscape, public datasets and object coverage, model use in application and algorithm & method studies, and the training & transfer analysis (966 included papers).

- Entry point: `app/app.py`
- Data: `app/data/` (paper evidence snapshot, English display text in `en.json`) and `project/` (official list, wave-2 extraction, dataset table)
- Display language: English. Internal data keep the original coding; `app/i18n.py` translates everything shown on screen.

Run locally:
```
pip install -r requirements.txt
streamlit run app/app.py
```
