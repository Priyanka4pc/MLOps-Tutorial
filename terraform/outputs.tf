output "project_id" {
    value = var.project_id
}

output "region"{
    value = var.region
}

output "gcs_dataset_bucket" {
  value = google_storage_bucket.dataset_bucket.name
}

output "dataset_name" {
  value = google_bigquery_dataset.example_dataset.dataset_id
}

output "train_bq_table" {
  value = var.train_bq_table
}

output "test_bq_table" {
  value = var.test_bq_table
}

# output "fs_name" {
#   value = google_vertex_ai_featurestore.featurestore.name
# }

# output "fs_entity_name" {
#   value = google_vertex_ai_featurestore_entitytype.entity.name
# }

output "vertex_endpoint" {
  value = google_vertex_ai_endpoint.endpoint.id
}

output "service_account" {
  value = var.service_account
}
