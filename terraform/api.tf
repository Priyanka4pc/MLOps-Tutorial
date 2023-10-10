# Enable Cloud Dataproc API
resource "google_project_service" "dataproc" {
  project = var.project_id
  service = "dataproc.googleapis.com"
}

# Enable Cloud Dataproc Control API PRIVATE
# resource "google_project_service" "dataproc_control" {
#   project = var.project_id
#   service = "dataproc-control.googleapis.com"
# }

# Enable Cloud Resource Manager API
resource "google_project_service" "resource_manager" {
  project = var.project_id
  service = "cloudresourcemanager.googleapis.com"
}

# Enable Compute Engine API
resource "google_project_service" "compute_engine" {
  project = var.project_id
  service = "compute.googleapis.com"
}

# Enable Vertex AI API
resource "google_project_service" "vertex_ai" {
  project = var.project_id
  service = "aiplatform.googleapis.com"
}
