provider "google" {
  alias = "tokengen"
}
# get config of the client that runs
data "google_client_config" "default" {
  provider = google.tokengen

}
data "google_service_account_access_token" "sa" {
  provider               = google.tokengen
  target_service_account = "mlops-tutorial@mlops-project-401617.iam.gserviceaccount.com"
  lifetime               = "600s"
  scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
  ]
}

/******************************************
  GA Provider configuration
 *****************************************/
provider "google" {
  access_token = data.google_service_account_access_token.sa.access_token
  project      = var.project_id
}

# define terraform provider
terraform {
  backend "gcs" {
    bucket = "tf-state-prod-mlops"
    prefix = "terraform/state"
  }
}

# provider "google" {
#   project = var.project_id
#   # credentials = file(var.creds_file)  # required when terraform authenticates on it's own
#   impersonate_service_account = "mlops-tutorial@mlops-project-401617.iam.gserviceaccount.com"
#   region = var.region
#   zone   = var.zone
# }