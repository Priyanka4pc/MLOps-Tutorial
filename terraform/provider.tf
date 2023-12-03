# define terraform provider
terraform {
  backend "gcs" {
    bucket = "tf-state-mlops-tut"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region = var.region
  zone   = var.zone
}