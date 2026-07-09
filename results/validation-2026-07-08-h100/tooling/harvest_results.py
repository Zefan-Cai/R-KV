"""Harvest sweep results from results/logs/*.log into one JSON summary.

Each benchmark log ends with a machine-readable JSON line; progress.log has
per-run status. Prints a JSON document: {runs: {name: {...}}, failures: [...]}.
Run on the node: venv/bin/python harvest_results.py /mnt/localssd/zefan/rkv-fi/results
"""
import json
import re
import sys
from pathlib import Path

results = Path(sys.argv[1])
runs: dict[str, dict] = {}
failures: list[str] = []

status: dict[str, str] = {}
for line in (results / "progress.log").read_text().splitlines():
    m = re.match(r"DONE (gpu\d) (\S+) (\S+) (\d+)s", line)
    if m:
        status[m.group(2)] = f"{m.group(3)} {m.group(4)}s {m.group(1)}"

for log in sorted((results / "logs").glob("*.log")):
    name = log.stem
    payload = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                pass
    entry: dict = {"status": status.get(name, "?")}
    if payload:
        entry["summary"] = payload
    else:
        for marker in ("SMOKE_PASS", "SMOKE_FAIL", "COMPARE_HF_PASS", "COMPARE_HF_FAIL",
                       "PROBE_OK", "PROBE_SUSPECT"):
            if marker in log.read_text(errors="replace"):
                entry["marker"] = marker
                break
    runs[name] = entry
    if "FAIL" in entry["status"] or entry.get("marker", "").endswith(("FAIL", "SUSPECT")):
        failures.append(name)

print(json.dumps({"runs": runs, "failures": failures}, indent=1))
