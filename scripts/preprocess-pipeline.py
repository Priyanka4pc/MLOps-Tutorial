import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from kfp.dsl import component
from google.cloud import aiplatform

load_dotenv("/workspace/.env")

PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET = os.environ.get("BQ_DATASET")
SRC_TABLE = os.environ.get("TRAIN_BQ_TABLE")
DATA_BUCKET = os.environ.get("DATA_BUCKET")


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
def preprocess(project_id: str, gcs_bucket: str, dataset_name: str, table_name: str):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from google.cloud import bigquery

    # Load the CSV data into a pandas DataFrame
    data = pd.read_csv(f"gs://{gcs_bucket}/train_data.csv")

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
    table = f"{project_id}.{dataset_name}.{table_name}"
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    job = client.load_table_from_dataframe(data, table, job_config=job_config)
    job.result()


@dsl.pipeline(
    name="Preprocess pipeline",
    description="A pipeline that preprocess data and stores it in BigQuery",
)
def preprocess_pipeline(
    project_id: str = PROJECT_ID,
    data_bucket: str = DATA_BUCKET,
    bq_dataset: str = DATASET,
    bq_table: str = SRC_TABLE,
):
    preprocess(
        project_id=project_id,
        gcs_bucket=data_bucket,
        dataset_name=bq_dataset,
        table_name=bq_table,
    )


if __name__ == "__main__":
    pipeline_file_name = "preprocess_pipeline.yaml"
    compiler.Compiler().compile(preprocess_pipeline, pipeline_file_name)

    pipeline_job = aiplatform.PipelineJob(
        display_name=f"PreprocessPipeline",
        template_path=pipeline_file_name,
        enable_caching=False,
    )

    response = pipeline_job.submit(service_account=os.environ.get("SERVICE_ACCOUNT"))
    pipeline_job.wait()
