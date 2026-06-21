import asyncio, aiohttp, time, json, argparse, statistics

FIXED_PROMPT = (
    "You are a helpful assistant. Summarize the history of computing in detail, "
    "covering mechanical calculators, vacuum tubes, transistors, integrated circuits, "
    "the microprocessor, personal computers, the internet, mobile, and modern AI. "
    "Be thorough and structured. Begin your detailed summary now: "
)


async def get_model(session, base):
    async with session.get(base + "/v1/models") as r:
        d = await r.json()
        return d["data"][0]["id"]


async def one_request(session, url, model, max_tokens):
    payload = {
        "model": model,
        "prompt": FIXED_PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = None
    token_times = []
    n = 0
    async with session.post(url, json=payload) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            now = time.perf_counter()
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            ch = chunk.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)
                n += 1
    return {"ttft": ttft, "token_times": token_times, "n": n, "total": time.perf_counter() - t0}


async def run_level(base, model, max_tokens, c, rounds):
    url = base + "/v1/completions"
    conn = aiohttp.TCPConnector(limit=0)
    to = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(connector=conn, timeout=to) as s:
        await asyncio.gather(*[one_request(s, url, model, max_tokens) for _ in range(c)])
        results, t0 = [], time.perf_counter()
        for _ in range(rounds):
            results += await asyncio.gather(*[one_request(s, url, model, max_tokens) for _ in range(c)])
        wall = time.perf_counter() - t0
    return results, wall


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def agg(results, wall, c):
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    tpots, itls = [], []
    for r in results:
        if r["n"] > 1 and r["ttft"] is not None:
            tpots.append((r["total"] - r["ttft"]) / (r["n"] - 1))
        tt = r["token_times"]
        itls += [tt[i] - tt[i - 1] for i in range(1, len(tt))]
    tot_tok = sum(r["n"] for r in results)
    return {
        "concurrency": c,
        "n_req": len(results),
        "qps": round(len(results) / wall, 3),
        "out_tok_s": round(tot_tok / wall, 1),
        "ttft_p50_ms": round(1000 * pct(ttfts, 50), 1) if ttfts else None,
        "ttft_p95_ms": round(1000 * pct(ttfts, 95), 1) if ttfts else None,
        "tpot_mean_ms": round(1000 * statistics.mean(tpots), 2) if tpots else None,
        "itl_p95_ms": round(1000 * pct(itls, 95), 2) if itls else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32")
    a = ap.parse_args()

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as s:
        model = a.model or await get_model(s, a.base)
    print(f"[{a.tag}] model={model} osl={a.osl} rounds={a.rounds}")

    rows = []
    for c in [int(x) for x in a.concurrencies.split(",")]:
        results, wall = await run_level(a.base, model, a.osl, c, a.rounds)
        row = agg(results, wall, c)
        rows.append(row)
        print(f"  c={c:>3}  QPS={row['qps']:>6}  out_tok/s={row['out_tok_s']:>7}  "
              f"TTFT_p50={row['ttft_p50_ms']}ms  TPOT={row['tpot_mean_ms']}ms  ITL_p95={row['itl_p95_ms']}ms")

    out = {"tag": a.tag, "model": model, "osl": a.osl, "rounds": a.rounds, "rows": rows}
    with open(f"results_{a.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> results_{a.tag}.json")


if __name__ == "__main__":
    asyncio.run(main())
