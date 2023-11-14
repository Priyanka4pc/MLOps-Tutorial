# define terraform provider
terraform {
  backend "gcs" {
    bucket = "tf-state-mlops"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  # credentials = file(var.creds_file)  # required when terraform authenticates on it's own
  region = var.region
  zone   = var.zone
}