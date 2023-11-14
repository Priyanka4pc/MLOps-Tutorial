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

# create a big query dataset 
resource "google_bigquery_dataset" "example_dataset" {
  dataset_id                 = "dataset_${var.suffix}"
  project                    = var.project_id
  delete_contents_on_destroy = true
  labels = {
    environment = "development"
  }
}

# create a feature store
# resource "google_vertex_ai_featurestore" "featurestore" {
#   name   = "feature_store_${var.suffix}"
#   region = var.region
#   online_serving_config {
#     fixed_node_count = 1
#   }
#   force_destroy = true
# }

# resource "google_vertex_ai_featurestore_entitytype" "entity" {
#   name = "data_features"

#   description  = "house features"
#   featurestore = google_vertex_ai_featurestore.featurestore.id

#   monitoring_config {
#     snapshot_analysis {
#       disabled                 = false
#       monitoring_interval_days = 1
#       staleness_days           = 21
#     }
#     numerical_threshold_config {
#       value = 0.8
#     }
#     categorical_threshold_config {
#       value = 10.0
#     }
#     import_features_analysis {
#       state                      = "ENABLED"
#       anomaly_detection_baseline = "PREVIOUS_IMPORT_FEATURES_STATS"
#     }
#   }
# }

resource "google_vertex_ai_endpoint" "endpoint" {
  name         = "sklearn-endpoint"
  display_name = "sklearn-endpoint"
  description  = "Model endpoint"
  location     = var.region
  region       = var.region

}
resource "google_pubsub_topic" "monitoring_topic" {
  name = "monitoring-topic"
  message_retention_duration = "86600s"
}

resource "google_monitoring_notification_channel" "basic" {
  display_name = "Model Monitoring Notification Channel"
  type = "pubsub"
  labels = {
     topic = "projects/${var.project_id}/topics/${google_pubsub_topic.monitoring_topic.name}"
   }
}
