import os
import datetime
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import bigquery
from google.cloud.aiplatform import Featurestore
from google.cloud.bigquery.table import TableReference

load_dotenv("/workspace/.env")

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
DATASET = os.environ.get("BQ_DATASET")
SRC_TABLE = os.environ.get("BQ_TABLE")
FEATURE_STORE_NAME = os.environ.get("FS_NAME")
ENTITY_TYPE_ID = os.environ.get("FS_ENTITY_NAME")

BQ_CLIENT_INFO = ClientOptions(quota_project_id=PROJECT_ID)
BQ_CLIENT = bigquery.client.Client(project=PROJECT_ID, client_options=BQ_CLIENT_INFO)


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
    table_obj = BQ_CLIENT.get_table("{}.{}.{}".format(PROJECT_ID, DATASET, SRC_TABLE))
    for s in table_obj.schema:
        if s.name.lower() not in ["index_column"]:
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
        if s.name.lower() not in ["index_column"]
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
            feature_time=datetime.datetime.now(),
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
