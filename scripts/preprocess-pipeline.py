import os
from dotenv import load_dotenv
from kfp import dsl, compiler
from kfp.dsl import component
from google.cloud import aiplatform

load_dotenv("/workspace/.env")

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
DATASET = os.environ.get("BQ_DATASET")
SRC_TABLE = os.environ.get("BQ_TABLE")
DATA_BUCKET = os.environ.get("DATA_BUCKET")
FEATURE_STORE_NAME = os.environ.get("FS_NAME")
ENTITY_TYPE_ID = os.environ.get("FS_ENTITY_NAME")


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

    # Convert "Date" column to date type
    data["time"] = pd.to_datetime(data["Date"])

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

    data = data.reset_index()
    data = data.rename(columns={"index": "index_column"})
    data["index_column"] = data["index_column"].astype(str)

    # Ingest the preprocessed data into BigQuery
    # Upload data to BigQuery
    client = bigquery.Client(project=project_id)
    table = f"{project_id}.{dataset_name}.{table_name}"
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job_config.schema = [
        bigquery.SchemaField("time", bigquery.enums.SqlTypeNames.TIMESTAMP),
    ]

    job = client.load_table_from_dataframe(data, table, job_config=job_config)
    job.result()


@component(
    base_image="python:3.10",
    packages_to_install=[
        "python-dotenv",
        "google-api-core",
        "google-cloud-bigquery",
        "google-cloud-aiplatform",
    ],
)
def create_features(
    project_id: str,
    region: str,
    bq_dataset: str,
    bq_table: str,
    fs_name: str,
    fs_entity_name: str,
):
    from google.api_core.client_options import ClientOptions
    from google.cloud import bigquery
    from google.cloud.aiplatform import Featurestore
    from google.cloud.bigquery.table import TableReference

    PROJECT_ID = project_id
    REGION = region
    DATASET = bq_dataset
    SRC_TABLE = bq_table
    FEATURE_STORE_NAME = fs_name
    ENTITY_TYPE_ID = fs_entity_name

    BQ_CLIENT_INFO = ClientOptions(quota_project_id=PROJECT_ID)
    BQ_CLIENT = bigquery.client.Client(
        project=PROJECT_ID, client_options=BQ_CLIENT_INFO
    )

    def map_dtype_to_featurestore(feature_type):
        if feature_type == "STRING":
            return "STRING"
        elif feature_type in [
            "INTEGER",
            "INT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "TINYINT",
            "BYTEINT",
            "INT64",
        ]:
            return "INT64"
        elif feature_type.startswith("ARRAY") or feature_type.startswith("STRUCT"):
            raise "Cannot process source table having columns with datatype " + feature_type
        elif feature_type in [
            "BIGNUMERIC",
            "NUMERIC",
            "DECIMAL",
            "BIGDECIMAL",
            "FLOAT64",
            "FLOAT",
        ]:
            return "DOUBLE"
        elif (
            feature_type.startswith("BIGNUMERIC")
            or feature_type.startswith("NUMERIC")
            or feature_type.startswith("DECIMAL")
            or feature_type.startswith("BIGDECIMAL")
        ):
            return "DOUBLE"
        elif feature_type == "BOOL":
            return "BOOL"
        elif feature_type == "BYTES":
            return "BYTES"
        elif feature_type in ["DATE", "DATETIME", "INTERVAL", "TIME", "TIMESTAMP"]:
            return "STRING"
        elif feature_type == "JSON":
            return "STRING"
        else:
            return "STRING"

    def populate_feature_store(name):
        fs = Featurestore(
            featurestore_name=name,
            project=PROJECT_ID,
            location=REGION,
        )
        print(f"{fs.gca_resource=}")

        preprocessed_entity_type = fs.get_entity_type(entity_type_id=ENTITY_TYPE_ID)
        table_obj = BQ_CLIENT.get_table(
            "{}.{}.{}".format(PROJECT_ID, DATASET, SRC_TABLE)
        )
        for s in table_obj.schema:
            if s.name.lower() not in ["time", "index_column"]:
                preprocessed_entity_type.create_feature(
                    feature_id=s.name.lower(),
                    value_type=map_dtype_to_featurestore(s.field_type),
                )

        return fs, preprocessed_entity_type

    def get_feature_source_fields(preprocessed_entity_type):
        lof = preprocessed_entity_type.list_features(order_by="create_time")
        lofn = [f.name for f in lof]

        src_table = BQ_CLIENT.get_table(
            TableReference.from_string(
                "{}.{}.{}".format(PROJECT_ID, DATASET, SRC_TABLE),
                default_project=PROJECT_ID,
            )
        )
        columns = [
            s.name
            for s in src_table.schema
            if s.name.lower() not in ["time", "index_column"]
        ]

        print("Obtained mapping from feature store to bigquery")
        return lofn, dict(zip(lofn, columns))

    def populate_features_extract_features(fs, preprocessed_entity_type):
        try:
            lofn, feature_source_fields = get_feature_source_fields(
                preprocessed_entity_type
            )
            preprocessed_entity_type.ingest_from_bq(
                feature_ids=lofn,
                feature_time="time",
                bq_source_uri="bq://{}.{}.{}".format(PROJECT_ID, DATASET, SRC_TABLE),
                feature_source_fields=feature_source_fields,
                entity_id_field="index_column",
                sync=True,
            )
            print("Ingested Bigquery Source table into Feature Store")
        except:
            print("Error populating features in bigquery")
            raise

    fs, preprocessed_entity_type = populate_feature_store(FEATURE_STORE_NAME)
    populate_features_extract_features(fs, preprocessed_entity_type)


@dsl.pipeline(
    name="Preprocess pipeline",
    description="A pipeline that preprocess data and stores it in BigQuery and then into feature store",
)
def preprocess_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
    data_bucket: str = DATA_BUCKET,
    bq_dataset: str = DATASET,
    bq_table: str = SRC_TABLE,
    fs_name: str = FEATURE_STORE_NAME,
    fs_entity_name: str = ENTITY_TYPE_ID,
):
    preprocess_op = preprocess(
        project_id=project_id,
        gcs_bucket=data_bucket,
        dataset_name=bq_dataset,
        table_name=bq_table,
    )
    # create_features(
    #     project_id=project_id,
    #     region=region,
    #     bq_dataset=bq_dataset,
    #     bq_table=bq_table,
    #     fs_name=fs_name,
    #     fs_entity_name=fs_entity_name,
    # ).after(preprocess_op)


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
