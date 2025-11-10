import argparse, json, pathlib
def tok(s): return s.split()
def spans(toks, size, ov):
    i=0; n=len(toks)
    while i<n:
        j=min(i+size, n); yield toks[i:j]
        if j==n: break
        i=max(0, j-ov)
ap=argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--chunk_size", type=int, default=600)
ap.add_argument("--overlap", type=int, default=150)
a=ap.parse_args()
out=pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(a.inp, encoding="utf-8") as f_in, open(out,"w",encoding="utf-8") as f_out:
    for line in f_in:
        d=json.loads(line)
        tokens=tok(d["contents"])
        for idx, sp in enumerate(spans(tokens, a.chunk_size, a.overlap)):
            txt=" ".join(sp).strip()
            if not txt: continue
            f_out.write(json.dumps({"id": f'{d["id"]}_{idx}', "contents": txt}, ensure_ascii=False) + "\n")
print("Wrote chunks.jsonl")