import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from kfp.dsl import component, Output, Dataset
from google.cloud import aiplatform

load_dotenv("/workspace/.env")

PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET = os.environ.get("BQ_DATASET")
TEST_SRC_TABLE = os.environ.get("TEST_BQ_TABLE")
DATA_BUCKET = os.environ.get("DATA_BUCKET")
ENDPOINT = os.environ.get("ENDPOINT")


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery[pandas]",
        "pandas",
        "scikit-learn",
        "fsspec",
        "gcsfs"
    ],
)
def preprocess(project_id: str, gcs_bucket: str, dataset_name: str, test_table_name: str):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from google.cloud import bigquery

    # Load the CSV data into a pandas DataFrame
    data = pd.read_csv(f"gs://{gcs_bucket}/test_data.csv")

    # Preprocess the data
    data["CouncilArea"].fillna(data["CouncilArea"].mode()[0], inplace=True)
    data["YearBuilt"].fillna(data["YearBuilt"].mode()[0], inplace=True)
    data["BuildingArea"].fillna(data["BuildingArea"].mean(), inplace=True)

    # Convert categorical columns to numerical using LabelEncoder
    categorical_features = data.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for column in categorical_features:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Drop columns
    columns_to_drop = [
        "Address",
        "SellerG",
        "Suburb",
        "Date",
        "Lattitude",
        "Longtitude",
        "Postcode",
    ]
    data.drop(columns_to_drop, axis=1, inplace=True)

    # Convert column names to lowercase
    data.columns = data.columns.str.lower()

    # Ingest the preprocessed data into BigQuery
    # Upload data to BigQuery
    client = bigquery.Client(project=project_id)
    table = f"{project_id}.{dataset_name}.{test_table_name}"
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    job = client.load_table_from_dataframe(data, table, job_config=job_config)
    job.result()

@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery[pandas]==3.10.0",
        "pandas",
        "google-cloud-aiplatform"
    ],
)
def prediction(
    project_id: str,
    dataset_id: str,
    test_table_name: str,
    target_column: str,
    predictions_data: Output[Dataset],
    endpoint: str
):
    from google.cloud import bigquery
    import pandas as pd

    from google.cloud import aiplatform

    client = bigquery.Client(project=project_id)

    test_table = f"{project_id}.{dataset_id}.{test_table_name}"
    test_query = f"""
    SELECT * FROM {test_table}
    """
    job_config = bigquery.QueryJobConfig()
    test_query_job = client.query(query=test_query, job_config=job_config)
    test_df = test_query_job.result().to_dataframe()

    endpoint = aiplatform.Endpoint(endpoint_name=endpoint)
    test_X = test_df.drop(['price'], axis=1)
    
    predictions = []
    for features in test_X.values.tolist():
        resp = endpoint.predict([features])
        predictions.append(resp.predictions[0])
    df =  pd.DataFrame(test_df[target_column]).rename(columns={target_column: "Actual"})
    df["Predicted"]=predictions
    df["Predicted"] = df["Predicted"].round(2)
    df.to_csv(predictions_data.path, index=False)

@dsl.pipeline(
    name="Prediction pipeline",
    description="A pipeline that preprocess data and stores it in BigQuery and does predictions",
)
def prediction_pipeline(
    project_id: str = PROJECT_ID,
    data_bucket: str = DATA_BUCKET,
    bq_dataset: str = DATASET,
    test_bq_table: str = TEST_SRC_TABLE,
    target_column: str = "price"
):
    preprocess_op = preprocess(
        project_id=project_id,
        gcs_bucket=data_bucket,
        dataset_name=bq_dataset,
        test_table_name=test_bq_table,
    )

    prediction(
    project_id=project_id,
    dataset_id=bq_dataset,
    test_table_name=test_bq_table,
    endpoint=ENDPOINT,
    target_column=target_column
    ).after(preprocess_op)


if __name__ == "__main__":
    pipeline_file_name = "prediction_pipeline.yaml"
    compiler.Compiler().compile(prediction_pipeline, pipeline_file_name)

    pipeline_job = aiplatform.PipelineJob(
        display_name=f"PredictionPipeline",
        template_path=pipeline_file_name,
        enable_caching=False,
    )

    response = pipeline_job.submit(service_account=os.environ.get("SERVICE_ACCOUNT"))
    pipeline_job.wait()
