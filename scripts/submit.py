import argparse, csv, json, pathlib
from eval import read_questions, dummy_top5
ap=argparse.ArgumentParser()
ap.add_argument("--questions", required=True)
ap.add_argument("--faiss_dir", required=True)
ap.add_argument("--bm25_dir", required=True)
ap.add_argument("--k_retrieve", type=int, default=50)
ap.add_argument("--k_final", type=int, default=5)
ap.add_argument("--out", required=True)
a=ap.parse_args()
qs=read_questions(a.questions)
out=pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["q_id","web_list"])
    for qid,_ in qs: w.writerow([qid, json.dumps(dummy_top5(qid), ensure_ascii=False)])
print(f"Wrote {out}")