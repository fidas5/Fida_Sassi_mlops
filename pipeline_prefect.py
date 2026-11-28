import sys
import subprocess
import argparse
import importlib.util
import shutil

from pathlib import Path
from prefect import flow, task

from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_data,
)

# =====================================================
# 1. TASKS : MACHINE LEARNING
# =====================================================


@task(name="Load Dataset")
def task_load_data():
    data = load_data()

    print("Dataset loaded successfully.")

    return data


@task(name="Prepare Data")
def task_prepare_data(data):
    x_train, x_test, y_train, y_test = prepare_data(data)

    print("Data prepared successfully.")

    return x_train, x_test, y_train, y_test


@task(name="Train Model")
def task_train_model(x_train, y_train):
    model = train_model(x_train, y_train)

    print("Model trained successfully.")

    return model


@task(name="Save Model")
def task_save_model(model):
    save_model(model)

    print("Model saved successfully.")


@task(name="Load Model")
def task_load_model():
    model = load_model()

    print("Model loaded successfully.")

    return model


@task(name="Evaluate Model")
def task_evaluate_model(model, x_test, y_test):
    accuracy = evaluate_model(model, x_test, y_test)

    print("Model evaluated successfully.")

    return accuracy


@task(name="Predict Customer Churn")
def task_predict():
    results = predict_data()

    print("Customer churn predictions completed.")

    return results


# =====================================================
# 2. TASKS : CODE QUALITY
# =====================================================


@task(name="Format Code")
def task_format_code():
    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
        "test_model_pipeline.py",
    ]

    subprocess.run(
        [sys.executable, "-m", "black", *files],
        check=True,
    )

    print("Code formatted successfully.")


@task(name="Check Code Quality")
def task_check_quality():
    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
        "test_model_pipeline.py",
    ]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=88",
            "--extend-ignore=E203,W503",
            *files,
        ],
        check=True,
    )

    print("Code quality check passed.")


@task(name="Check Code Security")
def task_check_security():
    files = [
        "model_pipeline.py",
        "main.py",
        "pipeline_prefect.py",
    ]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-s",
            "B404,B603,B607",
            *files,
        ],
        check=True,
    )

    print("Security check passed.")


@task(name="Run Unit Tests")
def task_run_tests():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_model_pipeline.py",
            "-v",
        ],
        check=True,
    )

    print("Unit tests passed successfully.")


# =====================================================
# 3. TASKS : INSTALL AND CLEAN
# =====================================================


@task(name="Check Dependencies")
def task_install():
    packages = [
        "pandas",
        "numpy",
        "sklearn",
        "joblib",
        "prefect",
        "pytest",
        "black",
        "flake8",
        "bandit",
    ]

    missing = []

    for package in packages:
        if importlib.util.find_spec(package) is None:
            missing.append(package)

    if missing:
        raise RuntimeError(f"Missing dependencies: {missing}")

    print("All required dependencies are installed.")


@task(name="Clean Temporary Files")
def task_clean():
    project_dir = Path(__file__).resolve().parent

    for cache_dir in project_dir.rglob("__pycache__"):
        if "venv" not in cache_dir.parts and ".venv" not in cache_dir.parts:
            shutil.rmtree(cache_dir)

    pytest_cache = project_dir / ".pytest_cache"

    if pytest_cache.exists():
        shutil.rmtree(pytest_cache)

    print("Temporary Python cache files cleaned.")


@task(name="Sync Git Repository")
def task_sync_git():
    project_dir = Path(__file__).resolve().parent

    print("Checking Git repository...", flush=True)

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    if status.stdout.strip():
        print("Local changes detected: Git synchronization skipped.")
        return

    subprocess.run(
        ["git", "fetch", "origin", "main"],
        cwd=project_dir,
        check=True,
        timeout=120,
    )

    subprocess.run(
        ["git", "merge", "--ff-only", "origin/main"],
        cwd=project_dir,
        check=True,
        timeout=60,
    )

    print("Git repository synchronized successfully.")


# =====================================================
# 4. FLOW : TRAIN
# =====================================================


@flow(name="train")
def train_flow():
    # Step 1: Load data
    data = task_load_data()

    # Step 2: Prepare data
    x_train, x_test, y_train, y_test = task_prepare_data(data)

    # Step 3: Train model
    model = task_train_model(x_train, y_train)

    # Step 4: Save model for later prediction
    task_save_model(model)

    print("Training flow completed successfully!")

    return model


# =====================================================
# 5. FLOW : EVALUATE
# =====================================================


@flow(name="evaluate")
def evaluate_flow():
    # Step 1: Load data
    data = task_load_data()

    # Step 2: Prepare data
    x_train, x_test, y_train, y_test = task_prepare_data(data)

    # Step 3: Load saved model
    model = task_load_model()

    # Step 4: Evaluate model
    accuracy = task_evaluate_model(model, x_test, y_test)

    print("Evaluation flow completed successfully!")

    return accuracy


# =====================================================
# 6. FLOW : CODE
# =====================================================


@flow(name="code")
def code_flow():
    print("\n===== 1. CODE FORMATTING =====")
    task_format_code()

    print("\n===== 2. CODE QUALITY =====")
    task_check_quality()

    print("\n===== 3. SECURITY CHECK =====")
    task_check_security()

    print("\n===== 4. UNIT TESTING =====")
    task_run_tests()

    print("\n===== CODE FLOW COMPLETED SUCCESSFULLY =====")


# =====================================================
# 7. FLOW : ALL
# =====================================================


@flow(name="all")
def all_flow():
    # Step 1: Synchronize Git repository
    print("\n===== GIT SYNCHRONIZATION =====")
    task_sync_git()
    # Step 1: Code quality pipeline
    print("\n===== CODE QUALITY PIPELINE =====")
    code_flow()

    # Step 2: Load data
    print("\n===== DATA LOADING =====")
    data = task_load_data()

    # Step 3: Prepare data
    print("\n===== DATA PREPARATION =====")
    x_train, x_test, y_train, y_test = task_prepare_data(data)

    # Step 4: Train model
    print("\n===== MODEL TRAINING =====")
    model = task_train_model(x_train, y_train)

    # Step 5: Save model
    print("\n===== MODEL SAVING =====")
    task_save_model(model)

    # Step 6: Evaluate model
    print("\n===== MODEL EVALUATION =====")
    accuracy = task_evaluate_model(model, x_test, y_test)

    print("\n===== COMPLETE PIPELINE FINISHED - GITHUB VERSION =====")
    print(f"Final Accuracy: {accuracy:.4f}")

    return accuracy


# =====================================================
# 8. FLOW : PREDICT
# =====================================================


@flow(name="predict")
def predict_flow():
    print("\n===== CUSTOMER CHURN PREDICTION =====")

    results = task_predict()

    print("Prediction flow completed successfully!")

    return results


# =====================================================
# 9. FLOW : INSTALL
# =====================================================


@flow(name="install")
def install_flow():
    print("\n===== DEPENDENCIES CHECK =====")

    task_install()

    print("Install flow completed successfully!")


# =====================================================
# 10. FLOW : CLEAN
# =====================================================


@flow(name="clean")
def clean_flow():
    print("\n===== CLEAN TEMPORARY FILES =====")

    task_clean()

    print("Clean flow completed successfully!")


# =====================================================
# 11. MAIN : CHOOSE A FLOW
# =====================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Prefect Pipeline")

    parser.add_argument(
        "--flow",
        type=str,
        required=True,
        choices=[
            "all",
            "train",
            "evaluate",
            "code",
            "predict",
            "install",
            "clean",
        ],
        help="Choose the Prefect flow to execute",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()

    elif args.flow == "train":
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()

    elif args.flow == "code":
        code_flow()

    elif args.flow == "predict":
        predict_flow()

    elif args.flow == "install":
        install_flow()

    elif args.flow == "clean":
        clean_flow()
