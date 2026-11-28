import os
import hmac
import hashlib
import subprocess

from flask import Flask, request, abort

app = Flask(__name__)

WEBHOOK_SECRET = os.environ["GITHUB_WEBHOOK_SECRET"]

DEPLOYMENT = "all/ml-pipeline-all"


@app.route("/webhook", methods=["POST"])
def github_webhook():

    payload = request.get_data()
    signature = request.headers.get("X-Hub-Signature-256", "")

    expected_signature = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_signature):
        abort(403)

    event = request.headers.get("X-GitHub-Event")

    if event == "ping":
        return "Webhook connected", 200

    if event != "push":
        return "Event ignored", 200

    data = request.get_json()

    if data.get("ref") != "refs/heads/main":
        return "Branch ignored", 200

    subprocess.Popen(
        [
            "/home/fida/MLops_projet/venv/bin/prefect",
            "deployment",
            "run",
            DEPLOYMENT
        ],
        env={
            **os.environ,
            "PREFECT_API_URL": "http://127.0.0.1:4200/api"
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return "Prefect deployment trigger requested", 202


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
