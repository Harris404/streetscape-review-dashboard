"""Aggregate step-2 full-text extraction of the application group (509 papers): validate, normalise model names, write app_group_fulltext_extraction.csv and app_group_models_long.csv."""
import json, glob, csv, collections, re
C = {}
for f in sorted(glob.glob("ft_out/*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip(): d = json.loads(l); C[d["id"]] = d
exp = [json.loads(l)["id"] for f in glob.glob("ft_batches/*.jsonl") for l in open(f)]
print("records", len(C), "missing", sorted(set(exp) - set(C)), "extra", sorted(set(C) - set(exp)))
abs_ = {r["rec_id"]: r for r in csv.DictReader(open("app_group_abstract_extraction.csv", encoding="utf-8-sig"))}
RULES = [("mask2former", "Mask2Former"), ("maskformer", "MaskFormer"), ("mp-former", "MP-Former"), ("oneformer", "OneFormer"), ("segformer", "SegFormer"),
    ("segnext", "SegNeXt"), ("segmenter", "Segmenter"), ("internimage", "InternImage+UperNet"), ("upernet", "UperNet"), ("vit-adapter", "ViT-Adapter"), ("beit", "BEiT"), ("dpt", "DPT"),
    ("ovseg", "OVSeg"), ("semantic-sam", "SAM family"), ("grounded", "Grounded-SAM"), ("sam2", "SAM family"), ("segment anything", "SAM family"), (r"\bsam\b", "SAM family"),
    ("deeplabv3+", "DeepLabv3+"), ("deeplab v3+", "DeepLabv3+"), ("deeplabv3plus", "DeepLabv3+"), ("deeplab-v3+", "DeepLabv3+"), ("deeplab v3 +", "DeepLabv3+"),
    ("xception-71", "DeepLabv3+"), ("xception71", "DeepLabv3+"), ("ladeco", "DeepLabv3 (LaDeco)"), ("resnest", "DeepLabv3 (ResNeSt)"), ("deeplabv3", "DeepLabv3"), ("deeplab v3", "DeepLabv3"),
    ("deeplab-v3", "DeepLabv3"), ("deeplab", "DeepLab (unspecified)"), ("inplace-abn", "DeepLabv3+"), ("in-place abn", "DeepLabv3"), ("pspnet", "PSPNet"), ("ppm", "PSPNet (MIT CSAIL)"),
    ("mit csail", "PSPNet (MIT CSAIL)"), ("csail", "PSPNet (MIT CSAIL)"), ("hrnet", "HRNet(+OCR)"), ("ocrnet", "HRNet(+OCR)"), ("hierarchical multi-scale", "HMSA (NVIDIA)"),
    ("segnet", "SegNet"), ("fcn", "FCN"), ("fully conv", "FCN"), ("u-net", "U-Net"), ("unet", "U-Net"), (r"\be-?net\b", "ENet"), ("bisenet", "BiSeNet"), ("pp-liteseg", "PP-LiteSeg"),
    ("efficientvit", "EfficientViT"), ("dilated resnet", "Dilated ResNet (Wolfram)"), ("ademxapp", "Ademxapp (Wolfram)"), ("multi-scale context", "Multi-scale Context Agg."),
    ("panoptic fpn", "Panoptic FPN"), ("seamseg", "SeamSeg"), ("seamless", "SeamSeg"), ("mask r-cnn", "Mask R-CNN"), ("mask rcnn", "Mask R-CNN"), ("faster r-cnn", "Faster R-CNN"),
    ("fast r-cnn", "Fast R-CNN"), ("yolo", "YOLO"), ("ssd", "SSD"), ("grounding dino", "Grounding DINO"), ("detectron", "Detectron2 (unspecified)"), ("treepedia", "Treepedia"),
    ("swin", "Swin (head unspecified)"), ("unnamed", "unnamed")]
def norm(n):
    s = n.lower()
    for k, v in RULES:
        if (re.search(k, s) if k.startswith("\\") else k in s): return v
    return n
rows, long = [], []
for i, d in sorted(C.items()):
    a = abs_.get(i, {})
    ms = d.get("models") or []
    for m in ms:
        long.append({"rec_id": i, "model_raw": m.get("name", ""), "model": norm(m.get("name", "")), "backbone": m.get("backbone", ""), "task": m.get("task", ""),
                     "pretrained_on": "; ".join(m.get("pretrained_on") or []), "trained_by_authors": m.get("trained_by_authors", ""), "role": m.get("role", ""),
                     "theme": a.get("theme", ""), "year": a.get("year", "")})
    main = [m for m in ms if m.get("role") == "main"]
    rows.append({"rec_id": i, "confirm": d["confirm"], "code": d.get("code", ""), "main_models": "; ".join(sorted({norm(m["name"]) for m in main})),
                 "main_models_raw": "; ".join(m["name"] + (f" ({m['backbone']})" if m.get("backbone") else "") for m in main),
                 "pretrained_on": "; ".join(sorted({p for m in main for p in (m.get("pretrained_on") or [])})),
                 "trained_by_authors": "; ".join(sorted({m.get("trained_by_authors", "") for m in main})),
                 "comparison_models": "; ".join(sorted({norm(m["name"]) for m in ms if m.get("role") == "comparison"})),
                 "mentioned_models": "; ".join(sorted({norm(m["name"]) for m in ms if m.get("role") == "mentioned"})),
                 "classes_used": "; ".join(d.get("classes_used") or []), "indicators": "; ".join(d.get("indicators") or []), "framework": "; ".join(d.get("framework") or []),
                 "accuracy_check": d.get("accuracy_check", ""), "imagery_source": "; ".join(d.get("imagery_source") or []), "n_images": d.get("n_images", ""),
                 "theme": a.get("theme", ""), "location": a.get("location", ""), "evidence": d.get("evidence", ""), "title": a.get("title", ""), "year": a.get("year", ""), "doi": a.get("doi", "")})
for name, data in [("app_group_fulltext_extraction.csv", rows), ("app_group_models_long.csv", long)]:
    with open(name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
Cn = collections.Counter
print("confirm", dict(Cn(r["confirm"] for r in rows)), "| codes", dict(Cn(r["code"] for r in rows if r["code"])))
mainl = [x for x in long if x["role"] == "main"]
print("papers with >=1 named main model", sum(1 for r in rows if r["main_models"] and r["main_models"] != "unnamed"), "| unnamed only", sum(1 for r in rows if r["main_models"] == "unnamed"))
pm = Cn(); [pm.update({m}) for r in rows for m in set(r["main_models"].split("; ")) if m]
print("main model (papers)", pm.most_common(30))
print("weights (main)", Cn(p.strip() for x in mainl for p in x["pretrained_on"].split(";") if p.strip()).most_common(10))
print("trained_by_authors (main)", dict(Cn(x["trained_by_authors"] for x in mainl)))
print("comparison", Cn(m for r in rows for m in r["comparison_models"].split("; ") if m).most_common(8))
print("mentioned", Cn(m for r in rows for m in r["mentioned_models"].split("; ") if m).most_common(10))
print("framework", Cn(f.strip() for r in rows for f in r["framework"].split(";") if f.strip()).most_common(12))
print("accuracy_check none", sum(1 for r in rows if r["accuracy_check"].strip().lower().startswith("none") or not r["accuracy_check"].strip()), "of", len(rows))
