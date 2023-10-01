from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col

# Initialize a Spark session
spark = SparkSession.builder.appName("YourAppName").getOrCreate()

# Load the CSV file into a Spark DataFrame
data = spark.read.csv(destination_file_path, header=True, inferSchema=True)

# Define the target feature
TARGET_FEATURE = "Price"

# Select the target feature column
Y = data[TARGET_FEATURE]

# Separate numeric and categorical features
numeric_features = [col(column) for column, data_type in data.dtypes if data_type in ["int", "double"]]
categorical_features = [col(column) for column, data_type in data.dtypes if data_type == "string"]

# Impute missing values for categorical features
categorical_features = ["CouncilArea", "Type", "Method", "Regionname"]
for column in categorical_features:
    data = data.withColumn(column, when(data[column].isNull(), "Moreland").otherwise(data[column]))

# Impute missing values for numeric features
numeric_features_with_missing = [column for column in numeric_features if data.where(col(column).isNull()).count() > 0]
for column in numeric_features_with_missing:
    mean_value = data.select(mean(column)).collect()[0][0]
    data = data.withColumn(column, when(data[column].isNull(), mean_value).otherwise(data[column]))

# Encode categorical features using StringIndexer
indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(data) for column in categorical_features]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

# Create a list of training features
training_features = numeric_features + [column + "_index" for column in categorical_features if column != TARGET_FEATURE]

# Assemble the feature columns into a single vector column
assembler = VectorAssembler(inputCols=training_features, outputCol="features")
data = assembler.transform(data)

# Normalize the features using MinMaxScaler
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Select the final preprocessed data with scaled features and target column
data = data.select("scaled_features", TARGET_FEATURE)

# Save the preprocessed data as a CSV file
data.toPandas().to_csv("preprocessed.csv", index=False)

# Stop the Spark session
spark.stop()
