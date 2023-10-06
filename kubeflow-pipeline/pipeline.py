from typing import List
from kfp import dsl, compiler
from kfp.dsl import component, Input, Output, Metrics, Model, Artifact, Dataset

from google.cloud import aiplatform


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery[pandas]==3.10.0"],
)
def fetch_bigquery_dataset(
    project_id: str,
    dataset_id: str,
    view_name: str,
    dataset: Output[Dataset],
):
    """Exports from BigQuery to a CSV file.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.

    Returns:
        dataset: The Dataset artifact with exported CSV file.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    table_name = f"{project_id}.{dataset_id}.{view_name}"
    query = """
    SELECT
      *
    FROM
      `{table_name}`
    """.format(
        table_name=table_name
    )

    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.0.2", "pandas==1.3.5", "joblib==1.1.0"],
)
def train_model_op(model: Output[Model], accuracy: Output[Metrics]):
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import joblib

    data = pd.read_csv(
        "https://raw.githubusercontent.com/Priyanka4pc/ml-academy/main/preprocessed.csv"
    )

    X = data.drop(["Price"], axis=1)
    Y = data["Price"]

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=0)

    from sklearn.ensemble import RandomForestRegressor

    rfmodel = RandomForestRegressor(max_leaf_nodes=100, random_state=1)
    rfmodel.fit(train_X, train_Y)
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(rfmodel, os.path.join(model.path, "model.joblib"))

    score = rfmodel.score(test_X, test_Y)
    accuracy.log_metric("accuracy", score)


@component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_op(
    model: Input[Model],
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    from google.cloud import aiplatform

    aiplatform.init(project="gcp-tutorial-400612")

    deployed_model = aiplatform.Model.upload(
        display_name="rf-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
    )
    endpoint = deployed_model.deploy(machine_type="n1-standard-4")

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


@dsl.pipeline(
    name="Train multiple models",
    description="A pipeline that trains multiple models in parallel and selects best one to deploy",
)
def train_multiple_models_pipeline():
    train_op = train_model_op()
    deploy_model_op(model=train_op.outputs["model"])


if __name__ == "__main__":
    pipeline_file_name = "train_and_deploy_pipeline.yaml"
    compiler.Compiler().compile(train_multiple_models_pipeline, pipeline_file_name)

    pipeline_job = aiplatform.PipelineJob(
        display_name=f"test",
        template_path=pipeline_file_name,
        enable_caching=False,
    )

    response = pipeline_job.submit(
        service_account="gcp-tutorial@gcp-tutorial-400612.iam.gserviceaccount.com"
    )
