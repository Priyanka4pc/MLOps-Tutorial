variable "project_name" {
    default = "mlops-project-397918"
}

variable "creds_file" {
    default = "gcp-creds.json"
}

variable "region" {
    default = "us-central1"
}

variable "zone" {
    default = "us-central1-c"
}

variable "location" {
    default = "US"
}

variable "data_file" {
    default = "../data/train.csv"
}

variable "script_file" {
    default = "../scripts/outlier.py"
}