import json, glob, csv

files = sorted(glob.glob("results_*.json"))
if not files:
    raise SystemExit("no results_*.json found")

cols = ["qps", "out_tok_s", "ttft_p50_ms", "tpot_mean_ms", "itl_p95_ms"]
print(f"\n{'tag':<16}{'c':>4}{'QPS':>9}{'tok/s':>9}{'TTFTp50':>10}{'TPOT':>9}{'ITLp95':>9}")
print("-" * 66)
flat = []
for fn in files:
    d = json.load(open(fn))
    for r in d["rows"]:
        print(f"{d['tag']:<16}{r['concurrency']:>4}{r['qps']:>9}{r['out_tok_s']:>9}"
              f"{r['ttft_p50_ms']:>10}{r['tpot_mean_ms']:>9}{r['itl_p95_ms']:>9}")
        flat.append({"tag": d["tag"], "concurrency": r["concurrency"], **{k: r[k] for k in cols}})
    print("-" * 66)

with open("summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "concurrency"] + cols)
    w.writeheader()
    w.writerows(flat)
print("-> summary.csv")
