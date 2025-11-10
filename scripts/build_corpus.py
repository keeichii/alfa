import argparse, csv, json, pathlib, sys

# Поднять лимит поля (по умолчанию 131072 байт ≈ 128 КБ)
limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit = limit // 10

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
a = ap.parse_args()
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(a.inp, newline="", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
    r = csv.DictReader(f_in)
    for row in r:
        web_id = str(row.get("web_id","")).strip()
        title = (row.get("title") or "").strip()
        text  = (row.get("text") or "").strip()
        contents = (title + "\n" + text).strip()
        if not web_id or not contents: continue
        f_out.write(json.dumps({"id": web_id, "contents": contents}, ensure_ascii=False) + "\n")
print("Wrote corpus.jsonl")