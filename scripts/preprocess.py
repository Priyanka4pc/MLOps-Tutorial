import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from google.cloud import storage
# from google.cloud import bigquery
import os

# Set the path to your service account key file
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/prichoud3/Documents/MLOps-Tutorial/terraform/gcp-creds.json"
# # load data
# bucket_name = "train-data-9923"
# source_blob_name = "train.csv"

# Set the local file path where you want to save the downloaded data.
destination_file_path = "train.csv"

# storage_client = storage.Client()
# bucket = storage_client.bucket(bucket_name)
# blob = bucket.blob(source_blob_name)

# try:
#     # Download the blob to the specified file path.
#     blob.download_to_filename(destination_file_path)

#     print(f"Blob {source_blob_name} downloaded to {destination_file_path}")

# except Exception as e:
#     print(f"Error: {str(e)}")

data = pd.read_csv(destination_file_path)

TARGET_FEATURE = "Price"

Y = data[TARGET_FEATURE]

numeric_features = data.select_dtypes(["int", "float"]).columns
categorical_features = data.select_dtypes("object").columns

# impute missing values
data["CouncilArea"] = data["CouncilArea"].fillna("Moreland")
data["YearBuilt"] = data["YearBuilt"].fillna(data["YearBuilt"].mode()[0])
data["BuildingArea"] = data["BuildingArea"].fillna(data["BuildingArea"].mean())
data["Car"] = data["Car"].fillna(data["Car"].median())

# drop features with any unique values
categorical_features = categorical_features.drop("Address")
categorical_features = categorical_features.drop("SellerG")
categorical_features = categorical_features.drop("Suburb")

# define feature_columns the we convert to number
categorical_features = ["Type", "Method", "CouncilArea", "Regionname"]
categorical_features

for column in categorical_features:
    l_encoder = LabelEncoder()
    data[column] = l_encoder.fit_transform(data[column])

# create training features
training_features = list(numeric_features) + list(categorical_features)
training_features.remove("Price")

# normalize the training dataset
minMaxNorm = MinMaxScaler()
minMaxNorm.fit(data[training_features])
X = minMaxNorm.transform(data[training_features])
Y = data["Price"]

df_preprocessed = pd.concat([pd.DataFrame(X, columns=training_features),Y], axis=1)
df_preprocessed.to_csv("preprocessed.csv", index=False)

# store data to big query
# Set your Google Cloud project ID and BigQuery dataset ID and table ID.
# project_id = "mlops-project-39791"
# dataset_id = "dataset_9923"
# table_name = 'preprocessed_house_data'

# # Set the path to your local CSV file.
# csv_file_path = 'preprocessed.csv'

# client = bigquery.Client(project=project_id)

# dataset_ref = client.dataset(dataset_id)

# # Define the schema for your table (modify as needed).
# schema = [
#     bigquery.SchemaField("column1", "STRING"),
#     bigquery.SchemaField("column2", "INTEGER"),
#     # Add more fields as needed based on your CSV columns and their types.
# ]

# # Create the table with the defined schema.
# table_ref = dataset_ref.table(table_name)
# table = bigquery.Table(table_ref, schema=schema)
# table = client.create_table(table)  # Create the table

# print(f"Table {table_name} created.")

# # Load data into the newly created table.
# job_config = bigquery.LoadJobConfig(
#     autodetect=True,  # Automatically infer schema from the CSV
#     skip_leading_rows=1,  # Skip the CSV header
#     source_format=bigquery.SourceFormat.CSV,
# )
# try:
#     with open(csv_file_path, "rb") as source_file:
#         job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

#     job.result()  # Wait for the job to complete

#     print(f"Loaded {job.output_rows} rows into {table_name}")

# except Exception as e:
#     print(f"Error: {str(e)}")

