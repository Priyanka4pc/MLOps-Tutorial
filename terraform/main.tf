# create a bucket for storing the dataset
resource "google_storage_bucket" "dataset_bucket" {
  name          = "data-${var.suffix}"
  location      = var.region
  storage_class = "REGIONAL"
  force_destroy = true
}

# upload data to bucket
resource "google_storage_bucket_object" "csv_upload" {
  for_each = var.files
  name     = each.value
  source   = "${path.module}/${each.key}"
  bucket   = google_storage_bucket.dataset_bucket.name
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

resource "random_id" "vertex_endpoint_id" {
  byte_length = 4
}

resource "google_vertex_ai_endpoint" "endpoint" {
  name         = substr(random_id.vertex_endpoint_id.dec, 0, 10)
  display_name = "sklearn-endpoint"
  description  = "Model endpoint"
  location     = var.region
}

resource "google_pubsub_topic" "monitoring_topic" {
  name = "monitoring-topic"
  message_retention_duration = "86600s"
}

resource "google_pubsub_subscription" "example" {
  name  = "model-monitoring-subscription"
  topic = google_pubsub_topic.monitoring_topic.name

  # 20 minutes
  message_retention_duration = "200000s"
  retain_acked_messages      = true

  ack_deadline_seconds = 20

  expiration_policy {
    ttl = "300000s"
  }
  retry_policy {
    minimum_backoff = "10s"
  }

  enable_message_ordering = false
}

resource "google_monitoring_notification_channel" "basic" {
  display_name = "Model Monitoring Notification Channel"
  type = "pubsub"
  labels = {
     topic = "projects/${var.project_id}/topics/${google_pubsub_topic.monitoring_topic.name}"
   }
}

resource "google_storage_bucket" "bucket" {
  name     = "cloud-functions-bucket-${var.suffix}"
  location      = var.region
  storage_class = "REGIONAL"
  force_destroy = true
}

data "archive_file" "default" {
  type        = "zip"
  output_path = "/tmp/function-source.zip"
  source_dir  = "../scripts/retraining/"
}

resource "google_storage_bucket_object" "archive" {
  name   = "function-source.zip"
  bucket = google_storage_bucket.bucket.name
  source = data.archive_file.default.output_path # Path to the zipped function source code
}

resource "google_cloudfunctions2_function" "default" {
  name        = "ModelRetrainingFunction"
  location    = "us-central1"
  description = "Function to retrain model everytime drift is detected"

  build_config {
    runtime     = "python310"
    entry_point = "compile_pipeline" 

    source {
      storage_source {
        bucket = google_storage_bucket.bucket.name
        object = google_storage_bucket_object.archive.name
      }
    }
  }

  service_config {
    max_instance_count = 3
    min_instance_count = 1
    available_memory   = "256M"
    timeout_seconds    = 60
    all_traffic_on_latest_revision = true
    service_account_email          = var.service_account
    environment_variables = {
    PROJECT_ID = var.project_id,
    BQ_DATASET = google_bigquery_dataset.example_dataset.dataset_id,
    TEST_BQ_TABLE = var.test_bq_table,
    TRAIN_BQ_TABLE = var.train_bq_table,
    ENDPOINT = google_vertex_ai_endpoint.endpoint.id,
    SERVICE_ACCOUNT = var.service_account,
  }
  }

  event_trigger {
    trigger_region = "us-central1"
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.monitoring_topic.id
    retry_policy   = "RETRY_POLICY_DO_NOT_RETRY"
  }
}
