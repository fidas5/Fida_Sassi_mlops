from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import (
    all_flow,
    train_flow,
    predict_flow,
    evaluate_flow,
    install_flow,
    clean_flow,
    code_flow,
)

if __name__ == "__main__":

    # 1. Complete pipeline: every day at 02:00 Tunis time
    all_deployment = all_flow.to_deployment(
        name="ml-pipeline-all",
        schedule=Cron(
            "0 2 * * *",
            timezone="Africa/Tunis",
        ),
        tags=["mlops", "all"],
    )

    # 2. Model training: every Monday at 03:00 Tunis time
    train_deployment = train_flow.to_deployment(
        name="ml-pipeline-train",
        schedule=Cron(
            "0 3 * * 1",
            timezone="Africa/Tunis",
        ),
        tags=["mlops", "training"],
    )

    # 3. Predictions: every hour
    predict_deployment = predict_flow.to_deployment(
        name="ml-pipeline-predict",
        schedule=Cron(
            "0 * * * *",
            timezone="Africa/Tunis",
        ),
        tags=["mlops", "prediction"],
    )

    # 4. Evaluation: manual execution
    evaluate_deployment = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["mlops", "evaluation"],
    )

    # 5. Dependencies check: manual execution
    install_deployment = install_flow.to_deployment(
        name="ml-pipeline-install",
        tags=["mlops", "installation"],
    )

    # 6. Temporary files cleaning: manual execution
    clean_deployment = clean_flow.to_deployment(
        name="ml-pipeline-clean",
        tags=["mlops", "cleaning"],
    )

    # 7. Code quality: manual execution
    code_deployment = code_flow.to_deployment(
        name="ml-pipeline-code",
        tags=["mlops", "quality"],
    )

    # Register and serve all deployments
    serve(
        all_deployment,
        train_deployment,
        predict_deployment,
        evaluate_deployment,
        install_deployment,
        clean_deployment,
        code_deployment,
    )
