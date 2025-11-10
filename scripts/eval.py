import argparse, csv, json, pathlib, time
def read_questions(p):
    qs=[]; 
    with open(p, newline="", encoding="utf-8") as f:
        r=csv.reader(f)
        for row in r:
            if not row or not row[0].isdigit(): continue
            qs.append((int(row[0]), row[1]))
    return qs
def dummy_top5(q_id): return [1,2,3,4,5]
ap=argparse.ArgumentParser()
ap.add_argument("--questions", required=True)
ap.add_argument("--faiss_dir", required=True)
ap.add_argument("--bm25_dir", required=True)
ap.add_argument("--k_retrieve", type=int, default=50)
ap.add_argument("--k_final", type=int, default=5)
ap.add_argument("--report", required=True)
a=ap.parse_args()
qs=read_questions(a.questions)
# TODO: подключить реальный гибридный ретрив и реранк
res={qid: dummy_top5(qid) for qid,_ in qs}
report={
  "ts": int(time.time()),
  "k_retrieve": a.k_retrieve,
  "k_final": a.k_final,
  "notes": "TODO attach FAISS+BM25+CrossEncoder"
}
pathlib.Path(a.report).parent.mkdir(parents=True, exist_ok=True)
with open(a.report,"w",encoding="utf-8") as f: json.dump(report, f, ensure_ascii=False, indent=2)
print("Saved report")