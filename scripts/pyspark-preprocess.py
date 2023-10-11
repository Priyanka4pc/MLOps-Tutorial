import argparse

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import desc, mean, row_number, monotonically_increasing_id, col, to_utc_timestamp, to_date
from pyspark.sql.window import Window


def parse_arguments():
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    # Define positional arguments
    parser.add_argument("project", help="project id")
    parser.add_argument("dataset", help="dataset id")
    parser.add_argument("table", help="table name")
    parser.add_argument("bucket", help="bucket with training data")
    
    args = parser.parse_args()
    
    return args

def preprocessing(spark, gcs_bucket):

    # Load the CSV file into a Spark DataFrame
    data = spark.read.csv(f"gs://{gcs_bucket}/train_data.csv", header=True, inferSchema=True)

    # Select categorical features
    categorical_features = [t[0] for t in data.dtypes if t[1] == 'string']

    # Impute missing values
    data = data.fillna({'CouncilArea': 'Moreland'})

    mode_value = data.groupBy('YearBuilt').count().orderBy(desc('count')).collect()[1][0]
    data = data.fillna({'YearBuilt': mode_value})

    mean_value = data.select(mean(data['BuildingArea'])).collect()[0][0]
    data = data.fillna({'BuildingArea': mean_value})

    data = data.withColumn("Car", data["Car"].cast("float"))
    data = data.sort("Car")
    median_value = data.approxQuantile("Car", [0.5], 0)[0]
    data = data.fillna({'Car': median_value})

    data = data.withColumn("time", to_utc_timestamp(to_date(col("Date"), "d/M/yyyy"), "UTC"))

    # Define feature_columns that we convert to number
    categorical_features = ["Type", "Method", "CouncilArea", "Regionname"]

    for column in categorical_features:
        indexer = StringIndexer(inputCol=column, outputCol=column+"_index").fit(data)
        data = indexer.transform(data)

    data = data.dropna(subset=['Address'])
    data = data.drop(*categorical_features, "SellerG", "Suburb", "Date", "Lattitude", "Longtitude", "Postcode", "Address")

    for column in data.columns:
        data = data.withColumnRenamed(column, column.lower())

    # Add the index array as a new column
    data = data.withColumn("index_column", row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
    data = data.withColumn("index_column", data["index_column"].cast("string"))

    return data

def ingest_data_bq(data, args):

    project_id = args.project
    dataset_name = args.dataset
    table_name = args.table

    data.write.format("bigquery").option("temporaryGcsBucket", gcs_bucket).option("project", project_id).option(
        "dataset", dataset_name
    ).option("table", table_name).option(
        "createDisposition", "CREATE_IF_NEEDED"
    ).mode("overwrite").save()


if __name__ == "__main__":
     # Initialize a Spark session
    spark = SparkSession.builder.appName("MLOps").getOrCreate()

    args = parse_arguments()

    # Save the preprocessed data to BigQuery
    gcs_bucket = args.bucket
    data = preprocessing(spark, gcs_bucket)

    ingest_data_bq(data, args)
    
    # Stop the Spark session
    spark.stop()
