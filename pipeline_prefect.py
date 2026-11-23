from prefect import flow, task

import subprocess
import argparse
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


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


@task(name="Format Code")
def task_format_code():

    files = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

    subprocess.run(["black", *files], check=True)

    print("Code formatted successfully.")


@task(name="Check Code Quality")
def task_check_quality():

    files = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

    subprocess.run(
        ["flake8", "--max-line-length=88", "--extend-ignore=E203,W503", *files],
        check=True,
    )

    print("Code quality check passed.")


@task(name="Check Code Security")
def task_check_security():

    files = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

    subprocess.run(["bandit", "-s", "B404,B603,B607", *files], check=True)

    print("Security check passed.")


@task(name="Run Unit Tests")
def task_run_tests():

    subprocess.run(
        ["python", "-m", "pytest", "test_model_pipeline.py", "-v"], check=True
    )

    print("Unit tests passed successfully.")


@flow(name="train")
def train_flow():

    # Step 1: Load data
    data = task_load_data()

    # Step 2: Prepare data
    x_train, x_test, y_train, y_test = task_prepare_data(data)

    # Step 3: Train model
    model = task_train_model(x_train, y_train)

    print("Training flow completed successfully!")

    return model


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


@flow(name="all")
def all_flow():
    # Step 1: Code quality pipeline
    print("\n===== CODE QUALITY PIPELINE =====")
    code_flow()

    # Step 2: Load data
    print("\n===== DATA LOADING =====")
    data = task_load_data()

    # Step 3: Prepare data
    print("\n===== DATA PREPARATION =====")
    x_train, x_test, y_train, y_test = task_prepare_data(data)

    # Step 3: Train model
    print("\n===== 3. MODEL TRAINING =====")
    model = task_train_model(x_train, y_train)

    # Step 4: Save model
    print("\n===== 4. MODEL SAVING =====")
    task_save_model(model)

    # Step 5: Evaluate model
    print("\n===== 5. MODEL EVALUATION =====")
    accuracy = task_evaluate_model(model, x_test, y_test)

    print("\n===== COMPLETE PIPELINE FINISHED =====")
    print(f"Final Accuracy: {accuracy:.4f}")

    return accuracy


@flow(name="code")
def code_flow():

    # Step 1: Format code
    print("\n===== 1. CODE FORMATTING =====")
    task_format_code()

    # Step 2: Check code quality
    print("\n===== 2. CODE QUALITY =====")
    task_check_quality()

    # Step 3: Check security
    print("\n===== 3. SECURITY CHECK =====")
    task_check_security()

    # Step 4: Run unit tests
    print("\n===== 4. UNIT TESTING =====")
    task_run_tests()

    print("\n===== CODE FLOW COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Customer Churn Prefect Pipeline")

    parser.add_argument(
        "--flow",
        type=str,
        required=True,
        choices=["all", "train", "evaluate", "code"],
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
