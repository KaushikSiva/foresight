"""Fireworks operations with spend guards. Anything that costs money needs --yes."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://api.fireworks.ai/v1"
GPU_HOURLY = {"NVIDIA_H100_80GB": 8, "NVIDIA_H200_141GB": 8, "NVIDIA_B200_180GB": 13,
              "NVIDIA_B300_288GB": 15, "NVIDIA_GB300": 20}
DEFAULT_LIMIT = 8


def _request(method, path, body=None, data=None, headers=None):
    payload = json.dumps(body).encode() if body is not None else data
    request = urllib.request.Request(BASE + path, method=method, data=payload, headers={
        "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
        **({"Content-Type": "application/json"} if body is not None else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} failed: HTTP {e.code} {e.read()[:400].decode(errors='replace')}")


def account():
    return _request("GET", "/accounts")["accounts"][0]["name"]


def hourly(deployment):
    """Dollars per hour at one replica. None when the GPU type has no known price."""
    rate = GPU_HOURLY.get(deployment.get("acceleratorType"))
    return rate * deployment.get("acceleratorCount", 1) if rate else None


def status(_):
    deployments = _request("GET", f"/{account()}/deployments?pageSize=50").get("deployments", [])
    for d in deployments:
        print(d["name"], d.get("state"), d.get("baseModel"), d.get("acceleratorType"),
              d.get("acceleratorCount"), f"${hourly(d)}/hour", f"replicas={d.get('replicaCount')}")
    if not deployments:
        print("no active deployments")
    jobs = _request("GET", f"/{account()}/supervisedFineTuningJobs?pageSize=20").get("supervisedFineTuningJobs", [])
    for job in jobs:
        print(job["name"], job.get("state"), job.get("outputModel"), job.get("estimatedCost"), job.get("status", {}).get("message", ""))


def deploy(args):
    """Create at one replica, then delete immediately if it costs more than the limit."""
    if not args.yes:
        sys.exit(f"this starts billing up to ${args.max_hourly}/hour; pass --yes to proceed")
    deployment = _request("POST", f"/{account()}/deployments", {
        "displayName": args.name, "baseModel": args.model, "deploymentShape": "default",
        "minReplicaCount": 1, "maxReplicaCount": 1})
    cost = hourly(deployment)
    print(deployment["name"], deployment.get("acceleratorType"), deployment.get("acceleratorCount"), f"${cost}/hour")
    if cost is None or cost > args.max_hourly:
        _request("DELETE", f"/{deployment['name']}")
        sys.exit(f"deleted: cost ${cost}/hour is over the ${args.max_hourly}/hour limit")
    while True:
        current = _request("GET", f"/{deployment['name']}")
        print(time.strftime("%H:%M:%S"), current.get("state"), f"replicas={current.get('replicaCount')}", flush=True)
        if current.get("state") == "READY":
            print("READY", current["name"])
            return
        if current.get("state") in ("FAILED", "DELETING", "DELETED"):
            sys.exit(f"deployment ended as {current.get('state')}: {current.get('status')}")
        time.sleep(20)


def delete(args):
    _request("DELETE", f"/{args.name}")
    print("delete sent:", args.name)


def upload(args):
    """Create and upload a dataset. Storing it does not start any training."""
    path = Path(args.file)
    count = sum(1 for _ in open(path))
    acct = account()
    existing = [d["name"] for d in _request("GET", f"/{acct}/datasets?pageSize=200").get("datasets", [])]
    if f"{acct}/datasets/{args.dataset_id}" not in existing:
        _request("POST", f"/{acct}/datasets", {"datasetId": args.dataset_id, "dataset": {"userUploaded": {}, "exampleCount": str(count)}})
    boundary = "foresightboundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/jsonl\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    _request("POST", f"/{acct}/datasets/{args.dataset_id}:upload", data=body,
             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print("uploaded", args.dataset_id, count, "examples", _request("GET", f"/{acct}/datasets/{args.dataset_id}").get("state"))


def tune(args):
    if not args.yes:
        sys.exit("this starts a billed training job; pass --yes to proceed")
    acct = account()
    job = _request("POST", f"/{acct}/supervisedFineTuningJobs", {
        "displayName": args.output, "baseModel": args.base, "dataset": f"{acct}/datasets/{args.train}",
        "evaluationDataset": f"{acct}/datasets/{args.validation}", "outputModel": f"{acct}/models/{args.output}",
        "epochs": args.epochs, "loraRank": args.lora_rank})
    print(json.dumps({k: job.get(k) for k in ("name", "state", "outputModel", "epochs", "loraRank", "estimatedCost")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    sub.add_parser("status").set_defaults(fn=status)

    p = sub.add_parser("deploy")
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="foresight-planner")
    p.add_argument("--max-hourly", type=float, default=DEFAULT_LIMIT)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=deploy)

    p = sub.add_parser("delete")
    p.add_argument("name", help="full name, accounts/<id>/deployments/<id>")
    p.set_defaults(fn=delete)

    p = sub.add_parser("upload")
    p.add_argument("dataset_id")
    p.add_argument("file")
    p.set_defaults(fn=upload)

    p = sub.add_parser("tune")
    p.add_argument("--base", default="accounts/fireworks/models/gemma-3-12b-it")
    p.add_argument("--train", default="foresight-train")
    p.add_argument("--validation", default="foresight-validation")
    p.add_argument("--output", default="foresight-gemma-3-12b-planner")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=tune)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
