import os
import hmac
import hashlib
import subprocess
import threading

from pathlib import Path
from flask import Flask, request, abort

app = Flask(__name__)

WEBHOOK_SECRET = os.environ["GITHUB_WEBHOOK_SECRET"]

PROJECT_DIR = Path("/home/fida/MLops_projet")
DEPLOYMENT = "all/ml-pipeline-all"
REPOSITORY = "fidas5/Fida_Sassi_mlops"

# Empêche deux synchronisations simultanées
sync_lock = threading.Lock()


def run_command(command):
    return subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=True
    )


def sync_github():
    print("Vérification du repository...", flush=True)

    # Vérifier les changements locaux
    status = run_command(
        ["git", "status", "--porcelain"]
    )

    if status.stdout.strip():
        raise RuntimeError(
            "Synchronisation annulée : modifications locales détectées."
        )

    print("Récupération des modifications GitHub...", flush=True)

    run_command(
        ["git", "fetch", "origin", "main"]
    )

    print("Mise à jour du code local...", flush=True)

    run_command(
        ["git", "merge", "--ff-only", "origin/main"]
    )

    print("Synchronisation Git terminée !", flush=True)


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

    data = request.get_json(silent=True) or {}

    # Accepter uniquement notre repository
    repository = data.get("repository", {}).get("full_name")

    if repository != REPOSITORY:
        return "Repository ignored", 200

    # Accepter uniquement les push sur main
    if data.get("ref") != "refs/heads/main":
        return "Branch ignored", 200

    if data.get("deleted"):
        return "Deleted branch ignored", 200

    # Éviter les exécutions simultanées
    if not sync_lock.acquire(blocking=False):
        return "Synchronization already running", 409

    try:
        sync_github()

        print("Déclenchement du pipeline Prefect...", flush=True)

        subprocess.run(
            [
                str(PROJECT_DIR / "venv/bin/prefect"),
                "deployment",
                "run",
                DEPLOYMENT
            ],
            cwd=PROJECT_DIR,
            env={
                **os.environ,
                "PREFECT_API_URL": "http://127.0.0.1:4200/api"
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )

        print("Exécution Prefect demandée !", flush=True)

        return "Git synchronized and Prefect triggered", 202

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            RuntimeError) as error:

        print(f"Erreur : {error}", flush=True)
        return "Synchronization or Prefect trigger failed", 500

    finally:
        sync_lock.release()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
