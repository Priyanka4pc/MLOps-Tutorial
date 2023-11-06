import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from kfp.dsl import component, Input, Output, Metrics, Model, Artifact, Dataset

from google.cloud import aiplatform

load_dotenv("/workspace/.env")


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery[pandas]==3.10.0",
        "google-cloud-aiplatform",
        "google-cloud-bigquery-storage",
        "pyarrow",
    ],
)
def fetch_features(
    project_id: str,
    dataset_id: str,
    table_name: str,
    region: str,
    feature_store_name: str,
    fs_entity_name: str,
    entity_id_column: str,
    time_column: str,
    dataset: Output[Dataset],
):
    from google.cloud import bigquery
    from google.cloud.aiplatform import Featurestore

    client = bigquery.Client(project=project_id)

    table = f"{project_id}.{dataset_id}.{table_name}"
    query = f"""
    SELECT
        {entity_id_column}, {time_column}
    FROM
        {table}
    """
    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df = df.rename(columns={entity_id_column: fs_entity_name, time_column: "timestamp"})

    fs = Featurestore(
        featurestore_name=feature_store_name,
        project=project_id,
        location=region,
    )
    features = fs.batch_serve_to_df(
        read_instances_df=df, serving_feature_ids={fs_entity_name: ["*"]}
    )
    features = features.drop([f"entity_type_{fs_entity_name}", "timestamp"], axis=1)
    features.to_csv(dataset.path, index=False)


@component(
    base_image="python:3.10",
    packages_to_install=["scikit-learn==1.0.2", "pandas==1.3.5", "joblib==1.1.0"],
)
def train_model_op(
    model_name: str,
    target_column: str,
    dataset: Input[Dataset],
    model: Output[Model],
    accuracy: Output[Metrics],
):
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import joblib

    with open(dataset.path, "r") as train_data:
        data = pd.read_csv(train_data)

    X = data.drop([target_column], axis=1)
    Y = data[target_column]

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=0)
    if model_name == "DecisionTreeRegressor":
        from sklearn.tree import DecisionTreeRegressor

        regmodel = DecisionTreeRegressor(max_depth=5, random_state=1)

    elif model_name == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor

        regmodel = RandomForestRegressor(max_leaf_nodes=100, random_state=1)

    regmodel.fit(train_X, train_Y)
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(regmodel, os.path.join(model.path, f"{model_name}.joblib"))

    score = regmodel.score(test_X, test_Y)
    accuracy.log_metric("Accuracy", score)


@component(base_image="python:3.10")
def best_model(
    train1_accuracy: Input[Metrics],
    train2_accuracy: Input[Metrics],
    train1_model: Input[Model],
    train2_model: Input[Model],
    best_model: Output[Model],
):
    if train1_accuracy.metadata["Accuracy"] > train2_accuracy.metadata["Accuracy"]:
        best_model.uri = train1_model.uri
    else:
        best_model.uri = train2_model.uri


@component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_op(
    project_id: str,
    model_name: str,
    machine_type: str,
    serving_container_image_uri: str,
    model: Input[Model],
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    deployed_model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_container_image_uri,
    )
    endpoint = deployed_model.deploy(machine_type=machine_type)

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


@dsl.pipeline(
    name="Train and deploy pipeline",
    description="A pipeline that trains a model and deploys it to vertex",
)
def train_and_deploy_pipeline(
    project_id: str = os.environ.get("PROJECT_ID"),
    dataset_id: str = os.environ.get("BQ_DATASET"),
    table_name: str = os.environ.get("BQ_TABLE"),
    region: str = os.environ.get("REGION"),
    feature_store_name: str = os.environ.get("FS_NAME"),
    fs_entity_name: str = os.environ.get("FS_ENTITY_NAME"),
    entity_id_column: str = "index_column",
    time_column: str = "time",
    target_column: str = "price",
    model_name: str = "sklearn-model",
    machine_type: str = "n1-standard-2",
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
):
    fetch_data_op = fetch_features(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
        region=region,
        feature_store_name=feature_store_name,
        fs_entity_name=fs_entity_name,
        entity_id_column=entity_id_column,
        time_column=time_column,
    )
    with dsl.ParallelFor(
        models=["RandomForestRegressor", "DecisionTreeRegressor"], parallelism=1
    ) as model_name:
        train1 = train_model_op(
            model_name=model_name,
            target_column=target_column,
            dataset=fetch_data_op.outputs["dataset"],
        )
        train2 = train_model_op(
            model_name=model_name,
            target_column=target_column,
            dataset=fetch_data_op.outputs["dataset"],
        )

    best_model_op = best_model(
        train1.outputs["accuracy"],
        train1.outputs["model"],
        train2.outputs["accuracy"],
        train2.outputs["model"],
    )
    deploy_model_op(
        project_id=project_id,
        model_name=model_name,
        machine_type=machine_type,
        serving_container_image_uri=serving_container_image_uri,
        model=best_model_op.outputs["best_model"],
    )


if __name__ == "__main__":
    pipeline_file_name = "train_and_deploy_pipeline.yaml"
    compiler.Compiler().compile(train_and_deploy_pipeline, pipeline_file_name)

    pipeline_job = aiplatform.PipelineJob(
        display_name=f"Train&DeployPipeline",
        template_path=pipeline_file_name,
        enable_caching=False,
    )

    response = pipeline_job.submit(service_account=os.environ.get("SERVICE_ACCOUNT"))
