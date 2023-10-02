from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import desc, mean, to_date
from pyspark.sql.types import StructType, StructField

# Initialize a Spark session
spark = SparkSession.builder.appName("MLOps").getOrCreate()

# Load the CSV file into a Spark DataFrame
data = spark.read.csv("gs://train-data-011023/train.csv", header=True, inferSchema=True)

TARGET_FEATURE = "Price"

# Select numeric and categorical features
numeric_features = [t[0] for t in data.dtypes if t[1] == 'int' or t[1] == 'double']
categorical_features = [t[0] for t in data.dtypes if t[1] == 'string']

# Impute missing values
data = data.fillna({'CouncilArea': 'Moreland'})

mode_value = data.groupBy('YearBuilt').count().orderBy(desc('count')).collect()[1][0]
data = data.fillna({'YearBuilt': mode_value})

mode_value = data.groupBy('Date').count().orderBy(desc('count')).collect()[0][0]
data = data.fillna({'Date': mode_value})
data = data.withColumn('Date', to_date(data.Date, 'd/M/yyyy'))

mean_value = data.select(mean(data['BuildingArea'])).collect()[0][0]
data = data.fillna({'BuildingArea': mean_value})

data = data.withColumn("Car", data["Car"].cast("float"))
data = data.sort("Car")
median_value = data.approxQuantile("Car", [0.5], 0)[0]
data = data.fillna({'Car': median_value})


# Define feature_columns that we convert to number
categorical_features = ["Type", "Method", "CouncilArea", "Regionname"]

for column in categorical_features:
    indexer = StringIndexer(inputCol=column, outputCol=column+"_index").fit(data)
    data = indexer.transform(data)

data = data.dropna(subset=['Address'])
# Select the required columns and save to CSV
df = data.select(["Date", "Address", *categorical_features, *numeric_features])

schema = StructType()
for field in df.schema.fields:
    schema.add(StructField(field.name, field.dataType, False))
final_df = spark.createDataFrame(df.rdd, schema)

# Save the preprocessed data to BigQuery
project_id = "gcp-tutorial-400612"
dataset_name = "dataset_011023"
table_name = "preprocessed_data"

final_df.write.format("bigquery").option("temporaryGcsBucket", "train-data-011023").option("project", project_id).option(
    "dataset", dataset_name
).option("table", table_name).option(
    "createDisposition", "CREATE_IF_NEEDED"
).mode("overwrite").save()

# Stop the Spark session
spark.stop()
