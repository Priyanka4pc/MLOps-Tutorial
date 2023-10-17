# create a bucket for storing the dataset
resource "google_storage_bucket" "dataset_bucket" {
  name          = "train-data-${var.suffix}"
  location      = var.region
  storage_class = "REGIONAL"
  force_destroy = true
}

# upload data to bucket
resource "google_storage_bucket_object" "csv_upload" {
  name         = "train_data.csv"
  bucket       = google_storage_bucket.dataset_bucket.name
  source       = var.data_file
  content_type = "text/csv"
}

# create a dataproc cluster
resource "google_dataproc_cluster" "preprocessing_cluster" {
  name    = "preprocessing-cluster-${var.suffix}"
  project = var.project_id
  region  = var.region
  cluster_config {
    master_config {
      num_instances = 1
      machine_type  = "n1-standard-2"
    }
    worker_config {
      num_instances = 2
      machine_type  = "n1-standard-2"
    }

  }
}

# create a big query dataset 
resource "google_bigquery_dataset" "example_dataset" {
  dataset_id                 = "dataset_${var.suffix}"
  project                    = var.project_id
  delete_contents_on_destroy = true
  labels = {
    environment = "development"
  }
}

# create a bucket to store preprocessing script for dataproc job
resource "google_storage_bucket" "script_bucket" {
  name          = "pyspark-script-${var.suffix}"
  location      = var.region
  storage_class = "REGIONAL"
  force_destroy = true
}

# uplaod preprocessing script to bucket
resource "google_storage_bucket_object" "python_script_upload" {
  name         = "pyspark-preprocess.py"
  bucket       = google_storage_bucket.script_bucket.name
  source       = var.script_file
  content_type = "text/x-python-script"
}

# run the preprocessing script in a dataproc job
resource "google_dataproc_job" "pyspark" {
  region       = google_dataproc_cluster.preprocessing_cluster.region
  force_delete = true
  placement {
    cluster_name = google_dataproc_cluster.preprocessing_cluster.name
  }

  pyspark_config {
    main_python_file_uri = "gs://${google_storage_bucket.script_bucket.name}/${google_storage_bucket_object.python_script_upload.name}"
    jar_file_uris        = ["gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"]
    args = [
      var.project_id,
      google_bigquery_dataset.example_dataset.dataset_id,
      var.bq_table,
      google_storage_bucket.dataset_bucket.name
    ]
    properties = {
      "spark.logConf" = "true"
    }
  }
  depends_on = [
    google_storage_bucket.script_bucket,
    google_storage_bucket_object.python_script_upload,
    google_bigquery_dataset.example_dataset,
    google_dataproc_cluster.preprocessing_cluster
  ]
}

# create a feature store
resource "google_vertex_ai_featurestore" "featurestore" {
  name   = "feature_store_${var.suffix}"
  region = var.region
  online_serving_config {
    fixed_node_count = 1
  }
  force_destroy = true
}

resource "google_vertex_ai_featurestore_entitytype" "entity" {
  name = "features"

  description  = "house features"
  featurestore = google_vertex_ai_featurestore.featurestore.id

  monitoring_config {
    snapshot_analysis {
      disabled                 = false
      monitoring_interval_days = 1
      staleness_days           = 21
    }
    numerical_threshold_config {
      value = 0.8
    }
    categorical_threshold_config {
      value = 10.0
    }
    import_features_analysis {
      state                      = "ENABLED"
      anomaly_detection_baseline = "PREVIOUS_IMPORT_FEATURES_STATS"
    }
  }
}
