# name of the project
variable "project_id" {
    default = "sustained-path-405106"
}

# name of the credentials file
variable "creds_file" {
    default = "gcp-creds.json"
}

# name of the region to use
variable "region" {
    default = "us-central1"
}

# name of the zone to use
variable "zone" {
    default = "us-central1-c"
}

# name of the location to use
variable "location" {
    default = "US"
}

# path to the data file
variable "data_file" {
    default = "../data/train_data.csv"
}

# path to the script file
# variable "script_file" {
#     default = "../scripts/pyspark-preprocess.py"
# }

variable "bq_table" {
    default = "preprocessed_data"
}

variable "suffix" {
    default = "141123"
}