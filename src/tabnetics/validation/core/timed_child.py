"""Run one command and record child-specific resource usage.

This small wrapper gives the shard runner per-job ``RUSAGE_CHILDREN`` metrics
from a fresh process, avoiding cumulative max-RSS leakage across jobs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabnetics.validation.core.provenance import (
    resource_usage_children,
    resource_usage_delta,
    utc_now_iso,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command and write resource-usage metrics.")
    parser.add_argument("--metrics-json", type=str, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = [str(item) for item in list(args.cmd or [])]
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("missing command after --")

    metrics_path = Path(args.metrics_json).expanduser().resolve()
    started_at = utc_now_iso()
    usage_start = resource_usage_children()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = int(proc.wait())
    usage_end = resource_usage_children()
    write_json(
        metrics_path,
        {
            "artifact_type": "tabnetics_validation_timed_child_resource_usage",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "exit_code": rc,
            "command": cmd,
            "resource_usage": {
                "children_start": usage_start,
                "children_end": usage_end,
                "children_delta": resource_usage_delta(usage_start, usage_end),
            },
        },
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
