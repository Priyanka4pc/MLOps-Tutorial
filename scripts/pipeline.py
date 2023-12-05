import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from typing import List
from kfp.dsl import component, Input, Output, Metrics, Model, Artifact, Dataset

from google.cloud import aiplatform

load_dotenv("/workspace/.env")


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery[pandas]==3.10.0",
    ],
)
def fetch_features(
    project_id: str,
    dataset_id: str,
    table_name: str,
    dataset: Output[Dataset],
):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    table = f"{project_id}.{dataset_id}.{table_name}"
    query = f"""
    SELECT * FROM {table}
    """
    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)

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
    joblib.dump(regmodel, os.path.join(model.path, "model.joblib"))

    score = regmodel.score(test_X, test_Y)
    accuracy.log_metric("Accuracy", score)


@component(base_image="python:3.10")
def best_model(
    model_inputs: Input[List[Model]],
    accuracy_inputs: Input[List[Metrics]],
    best_model: Output[Model],
):
    if (
        accuracy_inputs[0].metadata["Accuracy"]
        > accuracy_inputs[1].metadata["Accuracy"]
    ):
        best_model.uri = model_inputs[0].uri
    else:
        best_model.uri = model_inputs[1].uri


@component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model(
    project_id: str,
    model_name: str,
    endpoint: str,
    machine_type: str,
    target_column: str,
    serving_container_image_uri: str,
    model: Input[Model],
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types import SampledShapleyAttribution
    from google.cloud.aiplatform_v1.types.explanation import ExplanationParameters

    aiplatform.init(project=project_id)
    exp_metadata = {"inputs": {"Input_features": {}}, "outputs": {target_column: {}}}
    deployed_model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_container_image_uri,
        explanation_metadata=exp_metadata,
        explanation_parameters=ExplanationParameters(
            sampled_shapley_attribution=SampledShapleyAttribution(path_count=25)
        ),
    )
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint)
    endpoint = deployed_model.deploy(endpoint=endpoint, machine_type=machine_type)

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


@component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def model_monitoring(
    dataset_bq_uri: str,
    target: str,
    user_email: str,
    project_id: str,
    region: str,
    endpoint: Input[Artifact],
):
    from google.cloud.aiplatform import model_monitoring, ModelDeploymentMonitoringJob

    JOB_NAME = "mlops-tutorial"
    LOG_SAMPLE_RATE = 0.8
    MONITOR_INTERVAL = 5
    DATASET_BQ_URI = dataset_bq_uri
    TARGET = target

    skew_config = model_monitoring.SkewDetectionConfig(
        data_source=DATASET_BQ_URI,
        target_field=TARGET,
    )
    drift_config = model_monitoring.DriftDetectionConfig()
    explanation_config = model_monitoring.ExplanationConfig()
    objective_config = model_monitoring.ObjectiveConfig(
        skew_detection_config=skew_config,
        drift_detection_config=drift_config,
        explanation_config=explanation_config,
    )

    random_sampling = model_monitoring.RandomSampleConfig(sample_rate=LOG_SAMPLE_RATE)
    schedule_config = model_monitoring.ScheduleConfig(monitor_interval=MONITOR_INTERVAL)
    emails = [user_email]
    email_alert_config = model_monitoring.EmailAlertConfig(
        user_emails=emails, enable_logging=True
    )

    alerting_config = email_alert_config

    ModelDeploymentMonitoringJob.create(
        display_name=JOB_NAME,
        logging_sampling_strategy=random_sampling,
        schedule_config=schedule_config,
        alert_config=alerting_config,
        objective_configs=objective_config,
        project=project_id,
        location=region,
        endpoint=endpoint.uri,
    )


@dsl.pipeline(
    name="Train and deploy pipeline",
    description="A pipeline that trains a model and deploys it to vertex",
)
def train_and_deploy_pipeline(
    project_id: str = os.environ.get("PROJECT_ID"),
    dataset_id: str = os.environ.get("BQ_DATASET"),
    table_name: str = os.environ.get("TRAIN_BQ_TABLE"),
    region: str = os.environ.get("REGION"),
    endpoint: str = os.environ.get("ENDPOINT"),
    target_column: str = "price",
    model_name: str = "sklearn-model",
    machine_type: str = "n1-standard-4",
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
):
    fetch_data_op = fetch_features(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
    )
    with dsl.ParallelFor(
        items=["RandomForestRegressor", "DecisionTreeRegressor"], parallelism=2
    ) as item:
        train = train_model_op(
            model_name=item,
            target_column=target_column,
            dataset=fetch_data_op.outputs["dataset"],
        )

    best_model_op = best_model(
        model_inputs=dsl.Collected(train.outputs["model"]),
        accuracy_inputs=dsl.Collected(train.outputs["accuracy"]),
    )

    deploy_model_op = deploy_model(
        project_id=project_id,
        model_name=model_name,
        endpoint=endpoint,
        machine_type=machine_type,
        target_column=target_column,
        serving_container_image_uri=serving_container_image_uri,
        model=best_model_op.outputs["best_model"],
    )

    model_monitoring(
        project_id=project_id,
        dataset_bq_uri=f"bq://{project_id}.{dataset_id}.{table_name}",
        user_email="priyanka4iitd@gmail.com",
        target=target_column,
        region=region,
        endpoint=deploy_model_op.outputs["vertex_endpoint"],
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
    pipeline_job.wait()
