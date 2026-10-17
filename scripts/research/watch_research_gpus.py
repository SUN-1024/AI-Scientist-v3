#!/usr/bin/env python3
"""Sample only the assigned GPUs while one exact Linux process remains alive."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import time


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, IndexError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--uuids", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    identity = process_identity(args.pid)
    if identity is None:
        raise SystemExit("Research process is not alive")
    uuids = args.uuids.split(",")
    if len(uuids) != 2 or len(set(uuids)) != 2 or not all(x.startswith("GPU-") for x in uuids):
        raise SystemExit("Two distinct assigned GPU UUIDs are required")
    command = ["nvidia-smi","-i",args.uuids,
               "--query-gpu=uuid,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    status = {"pid":args.pid,"process_starttime":identity,"csv":str(args.csv),
              "gpu_uuids":uuids,"samples":0,"last_error":None}

    def save(state):
        status.update(status=state,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
        temporary = args.state.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.state)

    with args.csv.open("x", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp","gpu_uuid","memory_mib","utilization_percent"])
        while process_identity(args.pid) == identity:
            result = subprocess.run(command,capture_output=True,text=True,timeout=10)
            rows = [[x.strip() for x in row] for row in csv.reader(result.stdout.splitlines())]
            if result.returncode == 0 and {row[0] for row in rows if len(row) == 3} == set(uuids):
                now = datetime.now(timezone(timedelta(hours=8))).isoformat()
                for row in rows:
                    writer.writerow([now,*row])
                status["samples"] += 1
            else:
                status["last_error"] = "Assigned GPU query failed or returned a different allocation"
            save("sampling")
            time.sleep(1)
    save("finished")


if __name__ == "__main__":
    main()
