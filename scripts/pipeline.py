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
    target_column: str,
    dataset: Output[Dataset],
    analysis_schema: Output[Artifact],
):
    from google.cloud import bigquery

    # from google.cloud.aiplatform import Featurestore

    client = bigquery.Client(project=project_id)

    table = f"{project_id}.{dataset_id}.{table_name}"
    query = f"""
    SELECT * FROM {table}
    """
    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)

    bq_table = client.get_table(table)
    schema = bq_table.schema
    yaml = """type: object
properties:
"""
    for feature in schema:
        if feature.name == target_column:
            continue
        if feature.field_type == "STRING":
            f_type = "string"
        else:
            f_type = "integer"
        yaml += f"""    {feature.name}:
        type: {f_type}
"""

    yaml += """required:
    """
    for feature in schema:
        if feature.name == target_column:
            continue
        yaml += f"""- {feature.name}
    """
    with open(f"{analysis_schema.path}.yaml", "w") as f:
        f.write(yaml)


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


@component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform", "pyyaml"])
def deploy_model(
    project_id: str,
    model_name: str,
    endpoint: str,
    machine_type: str,
    serving_container_image_uri: str,
    target_column: str,
    schema: Input[Artifact],
    model: Input[Model],
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    from google.cloud import aiplatform
    import yaml

    with open(f"{schema.path}.yaml", "r") as stream:
        schema_dict = yaml.safe_load(stream)

    aiplatform.init(project=project_id)

    EXPLANATION_METADATA = aiplatform.explain.ExplanationMetadata(
        inputs={"input_features": {"encoding": "BAG_OF_FEATURES", "index_feature_mapping": schema_dict["required"]}}, outputs={target_column: {}}
    )
    PARAMETERS = {"sampled_shapley_attribution": {"path_count": 10}}
    parameters = aiplatform.explain.ExplanationParameters(PARAMETERS)

    deployed_model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_container_image_uri,
        explanation_parameters=parameters,
        explanation_metadata=EXPLANATION_METADATA,
    )
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint)
    endpoint = deployed_model.deploy(endpoint=endpoint, machine_type=machine_type)

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = deployed_model.resource_name


@component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform", "pyyaml"])
def model_monitoring(
    analysis_schema: Input[Artifact],
    dataset_bq_uri: str,
    target: str,
    user_email: str,
    project_id: str,
    region: str,
    endpoint: Input[Artifact],
):
    import yaml
    from google.cloud.aiplatform import model_monitoring, ModelDeploymentMonitoringJob

    JOB_NAME = "mlops-tutorial"

    # Sampling rate (optional, default=.8)
    LOG_SAMPLE_RATE = 0.8  # @param {type:"number"}

    # Monitoring Interval in hours (optional, default=1).
    MONITOR_INTERVAL = 1  # @param {type:"number"}

    # URI to training dataset.
    DATASET_BQ_URI = dataset_bq_uri  # @param {type:"string"}
    # Prediction target column name in training dataset.
    TARGET = target

    # # Skew and drift thresholds.

    DEFAULT_THRESHOLD_VALUE = 0.001

    with open(f"{analysis_schema.path}.yaml", "r") as stream:
        schema_dict = yaml.safe_load(stream)

    threshold = {
        feature: DEFAULT_THRESHOLD_VALUE for feature in schema_dict["required"]
    }
    SKEW_THRESHOLDS = threshold
    DRIFT_THRESHOLDS = threshold
    ATTRIB_SKEW_THRESHOLDS = threshold
    ATTRIB_DRIFT_THRESHOLDS = threshold

    skew_config = model_monitoring.SkewDetectionConfig(
        data_source=DATASET_BQ_URI,
        skew_thresholds=SKEW_THRESHOLDS,
        attribute_skew_thresholds=ATTRIB_SKEW_THRESHOLDS,
        target_field=TARGET,
    )

    drift_config = model_monitoring.DriftDetectionConfig(
        drift_thresholds=DRIFT_THRESHOLDS,
        attribute_drift_thresholds=ATTRIB_DRIFT_THRESHOLDS,
    )

    explanation_config = model_monitoring.ExplanationConfig()
    objective_config = model_monitoring.ObjectiveConfig(
        skew_detection_config=skew_config,
        drift_detection_config=drift_config,
        explanation_config=None,
    )

    # Create sampling configuration
    random_sampling = model_monitoring.RandomSampleConfig(sample_rate=LOG_SAMPLE_RATE)

    # Create schedule configuration
    schedule_config = model_monitoring.ScheduleConfig(monitor_interval=MONITOR_INTERVAL)

    # Create alerting configuration.
    emails = [user_email]
    email_alert_config = model_monitoring.EmailAlertConfig(
        user_emails=emails, enable_logging=True
    )

    alerting_config = email_alert_config
    # once notification channels is supported in model_monitoring
    # alerting_config = model_monitoring.AlertConfig(email_alert_config, enable_logging=True, notification_channels=["projects/mlops-project-401617/notificationChannels/13120585024516926294"])

    # Create the monitoring job.
    ModelDeploymentMonitoringJob.create(
        display_name=JOB_NAME,
        logging_sampling_strategy=random_sampling,
        schedule_config=schedule_config,
        alert_config=alerting_config,
        objective_configs=objective_config,
        project=project_id,
        location=region,
        endpoint=endpoint.uri,
        analysis_instance_schema_uri=f"{analysis_schema.uri}.yaml",
        predict_instance_schema_uri=f"{analysis_schema.uri}.yaml",
    )


@dsl.pipeline(
    name="Train and deploy pipeline",
    description="A pipeline that trains a model and deploys it to vertex",
)
def train_and_deploy_pipeline(
    project_id: str = os.environ.get("PROJECT_ID"),
    dataset_id: str = os.environ.get("BQ_DATASET"),
    table_name: str = os.environ.get("BQ_TABLE"),
    region: str = os.environ.get("REGION"),
    endpoint: str = os.environ.get("ENDPOINT"),
    target_column: str = "price",
    model_name: str = "sklearn-model",
    machine_type: str = "n1-standard-4",
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
):
    fetch_data_op = fetch_features(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
        target_column=target_column,
    )
    with dsl.ParallelFor(
        items=["RandomForestRegressor", "DecisionTreeRegressor"], parallelism=1
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
        schema=fetch_data_op.outputs["analysis_schema"],
        serving_container_image_uri=serving_container_image_uri,
        model=best_model_op.outputs["best_model"],
    )

    model_monitoring(
        project_id=project_id,
        analysis_schema=fetch_data_op.outputs["analysis_schema"],
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
