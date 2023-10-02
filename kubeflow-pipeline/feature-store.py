from kfp.v2.dsl import component

    
@component(
    packages_to_install=["pandas", "google-cloud-bigquery", "scikit-learn", "numpy", "google-cloud-aiplatform", "pyarrow", "google-cloud-storage"],
    base_image="python:3.10",
)
def run_feature_store_task(
    params: dict
):
    import datetime

    from google.api_core.client_options import ClientOptions
    from google.cloud import bigquery
    from google.cloud.aiplatform import Featurestore
    from google.cloud.bigquery.table import TableReference


    PROJECT_ID=params['PROJECT_ID']
    REGION=params['REGION']
    ONLINE_STORE_FIXED_NODE_COUNT=params['ONLINE_STORE_FIXED_NODE_COUNT']
    TEMP_DATASET=params['TEMP_DATASET']
    TEMP_SRC_TABLE=params['TEMP_SRC_TABLE']
    NEW_FEATURE_STORE_NAME=params['NEW_FEATURE_STORE_NAME']

    BQ_CLIENT_INFO = ClientOptions(quota_project_id = PROJECT_ID)
    BQ_CLIENT = bigquery.client.Client(project = PROJECT_ID, 
                                       client_options=BQ_CLIENT_INFO)

    def map_dtype_to_featurestore(feature_type):
        if feature_type == "STRING":
            return "STRING"
        elif feature_type in ["INTEGER", "INT", "SMALLINT", "INTEGER", "BIGINT", "TINYINT", "BYTEINT", "INT64"]:
            return "INT64"
        elif feature_type.startswith("ARRAY") or feature_type.startswith("STRUCT"):
            raise "Cannot process source table having columns with datatype " + feature_type
        elif feature_type in ["BIGNUMERIC", "NUMERIC", "DECIMAL", "BIGDECIMAL", "FLOAT64", "FLOAT"]:
            return "DOUBLE"
        elif feature_type.startswith("BIGNUMERIC") or feature_type.startswith("NUMERIC") or feature_type.startswith("DECIMAL") or feature_type.startswith("BIGDECIMAL"):
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
        fs_already_exists = False

        # Check if already exists
        try:
            fs = Featurestore(
                featurestore_name=name,
                project=PROJECT_ID,
                location=REGION,
            )
            print('Feature Store already exists')
            preprocessed_entity_type = fs.get_entity_type(
                entity_type_id="preprocessed"
            )
            fs_already_exists = True
            return fs, preprocessed_entity_type, fs_already_exists
        except Exception as e:
            print('Feature Store does not exists. Creating Feature Store')
            fs = Featurestore.create(
                featurestore_id=name,
                online_store_fixed_node_count=ONLINE_STORE_FIXED_NODE_COUNT,
                project=PROJECT_ID,
                location=REGION,
                sync=True,
            )

        fs = Featurestore(
            featurestore_name=name,
            project=PROJECT_ID,
            location=REGION,
        )
        print(f"{fs.gca_resource=}")

        preprocessed_entity_type = fs.create_entity_type(
            entity_type_id="preprocessed",
            description="Reading of metadata from app",
        )
        table_obj = BQ_CLIENT.get_table('{}.{}.{}'.format(PROJECT_ID, TEMP_DATASET, TEMP_SRC_TABLE))
        for s in table_obj.schema:
            preprocessed_entity_type.create_feature(
                feature_id=s.name.lower(),
                value_type=map_dtype_to_featurestore(s.field_type),
                # description="Unnamed integer column",
            )

        return fs, preprocessed_entity_type, fs_already_exists

    def get_feature_source_fields(preprocessed_entity_type):
        lof = preprocessed_entity_type.list_features(order_by='create_time')
        lofn = [f.name for f in lof]
        # LOGGER.info(lofn)

        src_table = BQ_CLIENT.get_table(TableReference.from_string('{}.{}.{}'.format(PROJECT_ID, TEMP_DATASET, TEMP_SRC_TABLE), default_project=PROJECT_ID))
        columns = [s.name for s in src_table.schema]

        print('Obtained mapping from feature store to bigquery')
        return lofn, dict(zip(lofn, columns))



    def populate_features_extract_features(fs, preprocessed_entity_type, fs_already_exists):
        try:
            lofn, feature_source_fields = get_feature_source_fields(preprocessed_entity_type)
            if fs_already_exists is False:
                preprocessed_entity_type.ingest_from_bq(
                    feature_ids=lofn,
                    feature_time=datetime.datetime.now(),
                    bq_source_uri='bq://{}.{}.{}'.format(PROJECT_ID, TEMP_DATASET, TEMP_SRC_TABLE),
                    feature_source_fields=feature_source_fields,
                    entity_id_field='reading_id',
                    disable_online_serving=False,
                    sync=True
                )
                print('Ingested Bigquery Source table into Feature Store')
        except:
            print('Error populating features in bigquery')
            raise

    fs, preprocessed_entity_type, fs_already_exists = populate_feature_store(NEW_FEATURE_STORE_NAME)
    populate_features_extract_features(fs, preprocessed_entity_type, fs_already_exists)