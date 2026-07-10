"""Manual check that the real Docker path works. Requires Docker + the built image:
    docker build -f Dockerfile.sandbox -t cadence-sandbox:latest .
    .venv/bin/python scripts/sandbox_smoke.py
Not part of CI (CI is service-free)."""
from agent.sandbox import run_in_sandbox

PROGRAM = (
    "import sys, json, pandas as pd\n"
    "d = json.load(sys.stdin)\n"
    "df = pd.DataFrame(d['rows'], columns=d['columns'])\n"
    "print(json.dumps({'n': int(len(df)), 'sum_x': int(df['x'].sum())}))\n"
)

if __name__ == "__main__":
    res = run_in_sandbox(PROGRAM, {"rows": [[1], [2], [3]], "columns": ["x"]})
    print("ok:", res.ok, "stdout:", res.stdout, "error:", res.error)
    assert res.ok and '"n": 3' in res.stdout and '"sum_x": 6' in res.stdout, res
    print("real-docker sandbox smoke passed (pandas available, stdin round-trip verified)")
