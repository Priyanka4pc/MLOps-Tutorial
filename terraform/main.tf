# define terraform provider
provider "google" {
  project     = var.project_name
  credentials = file(var.creds_file)
  region      = var.region
  zone        = var.zone
}

# create a bucket for storing the dataset
resource "google_storage_bucket" "dataset_bucket" {
  name          = "train-data-9923"
  location      = var.location
  force_destroy = true
}

# upload data to bucket
resource "google_storage_bucket_object" "csv_upload" {
  name         = "train.csv"
  bucket       = google_storage_bucket.dataset_bucket.name
  source       = "../data/train.csv"
  content_type = "text/csv"
}

# create a dataproc cluster
resource "google_dataproc_cluster" "preprocessing_cluster" {
  name    = "preprocessing-cluster"
  project = "mlops-project-397918"
  region  = var.region
  cluster_config {
    master_config {
      num_instances = 1
      machine_type  = "n1-standard-4"
    }
    worker_config {
      num_instances = 2
      machine_type  = "n1-standard-4"
    }

  }
}

# create a bucket to store preprocessing script for dataproc job
resource "google_storage_bucket" "script_bucket" {
  name     = "pyspark-script-9923"
  location = "US"
}

# included in milestone 2
# uplaod preprocessing script to bucket
# resource "google_storage_bucket_object" "python_script_upload" {
#   name   = "script.py" # Change to your desired object name
#   bucket = google_storage_bucket.script_bucket.name
#   source = "../scripts/outlier_detection.py" # Path to your local Python script
#   content_type = "text/plain" # Change the content type if needed
# }

# run the preprocessing script in a dataproc job
# resource "google_dataproc_job" "pyspark" {
#   region       = google_dataproc_cluster.preprocessing_cluster.region
#   force_delete = true
#   placement {
#     cluster_name = google_dataproc_cluster.preprocessing_cluster.name
#   }

#   pyspark_config {
#     main_python_file_uri = "gs://pyspark-script-9923/script.py"
#     properties = {
#       "spark.logConf" = "true"
#     }
#   }
# }

# create a big query dataset 
resource "google_bigquery_dataset" "example_dataset" {
  dataset_id = "dataset_9923"
  project    = "mlops-project-397918"
  labels = {
    environment = "development"
  }
}

# create a feature store
resource "google_vertex_ai_featurestore" "featurestore" {
  name   = "feature_store"
  region = var.region
  online_serving_config {
    fixed_node_count = 2
  }
  force_destroy = true
}
