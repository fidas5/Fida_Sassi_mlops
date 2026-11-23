from prefect import serve

from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow
)


if __name__ == "__main__":

    # Deployment for the complete pipeline
    all_deployment = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 8 * * *",
        tags=["mlops", "all"]
    )

    # Deployment for model training
    train_deployment = train_flow.to_deployment(
        name="ml-pipeline-train",
        cron="0 9 * * *",
        tags=["mlops", "training"]
    )

    # Deployment for model evaluation
    evaluate_deployment = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        cron="0 10 * * *",
        tags=["mlops", "evaluation"]
    )

    # Register and serve the deployments
    serve(
        all_deployment,
        train_deployment,
        evaluate_deployment
    )
