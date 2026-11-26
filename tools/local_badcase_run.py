#!/usr/bin/env python3
"""本地直连 hcOneshot /slowAgent（不经远端 runtime），把 csv 的 Query 列跑出 工具链/参数，供 judge_badcase.py 判定。
用法：python3 tools/local_badcase_run.py <in.csv> <out.csv> [--url http://127.0.0.1:8082] [--workers 6]"""
import argparse, csv, json, time, uuid, urllib.request
from concurrent.futures import ThreadPoolExecutor

def one(url, q):
    body = {"traceId": f"lb-{uuid.uuid4().hex[:12]}", "deviceId": f"lb-{uuid.uuid4().hex}", "deviceType": "VIDAA",
            "data": {"query": q, "memory": {"shortMemory": []}, "toolHistory": [], "toolList": [],
                     "tvMode": "0", "timestamp": int(time.time() * 1000)}}
    try:
        req = urllib.request.Request(f"{url}/slowAgent/poc_1", json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        steps = (d.get("data") or {}).get("steps") or []
        return " | ".join(s.get("toolName", "") for s in steps), json.dumps([s.get("parameters") or {} for s in steps], ensure_ascii=False), ""
    except Exception as e:  # noqa: BLE001
        return "", "", f"{type(e).__name__}: {e}"[:200]

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8082"); ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
rows = list(csv.DictReader(open(a.inp, encoding="utf-8-sig")))
with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(lambda r: one(a.url, r["Query"]), rows))
for r, (t, p, e) in zip(rows, res):
    r["工具链"], r["参数"], r["err"] = t, p, e
with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
