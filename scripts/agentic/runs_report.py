"""Summarize the agentic run database: expected vs. actual cost by model,
the latest runs, and plan forecasts.

    python scripts/agentic/runs_report.py                       # data/runs/agentic_runs.sqlite
    python scripts/agentic/runs_report.py --db runs.sqlite --last 20
    python scripts/agentic/runs_report.py --sql "SELECT * FROM v_call_costs WHERE cost_ratio > 2"
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=None, help="SQLite file (default: data/runs/agentic_runs.sqlite)")
    p.add_argument("--last", type=int, default=10, help="latest runs to list")
    p.add_argument("--sql", default=None, help="run this query instead of the report")
    args = p.parse_args()

    from router.tracing.run_store import connect

    conn = connect(args.db)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)
    if args.sql:
        print(pd.read_sql_query(args.sql, conn).to_string(index=False))
        return 0

    totals = pd.read_sql_query(
        "SELECT COUNT(*) AS runs, SUM(n_calls) AS calls, SUM(n_errors) AS errors, "
        "SUM(expected_cost) AS expected_usd, SUM(actual_cost) AS actual_usd FROM runs", conn)
    print("== totals\n" + totals.to_string(index=False))
    print("\n== expected vs actual cost by model (leaf calls)")
    print(pd.read_sql_query(
        "SELECT * FROM v_model_cost_accuracy ORDER BY n_calls DESC", conn).to_string(index=False))
    print(f"\n== last {args.last} runs")
    print(pd.read_sql_query(
        "SELECT recorded_at, mode, selected_model_id AS model, n_calls, n_errors, expected_cost, "
        "actual_cost, cost_ratio, total_s FROM v_run_costs ORDER BY recorded_at DESC LIMIT ?",
        conn, params=(args.last,)).to_string(index=False))
    plans = pd.read_sql_query("SELECT * FROM v_plan_costs ORDER BY plan_row_id DESC LIMIT ?", conn,
                              params=(args.last,))
    if len(plans):
        print(f"\n== last {args.last} plan revisions")
        print(plans.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
