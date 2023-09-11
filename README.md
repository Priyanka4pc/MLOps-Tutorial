# MLOps-Tutorial

## Milestone 1: Project Setup, Version Control, and Infrastructure as Code with Terraform

1. **Project Setup and Version Control:**
    - Create a new Google Cloud project.
      - Follow the steps to create a new project: https://console.cloud.google.com/projectcreate
      - Link a billing account to your google project
    - Create a service account:
      - Go to https://console.cloud.google.com/
      - Click on IAM & Admin
        ![Console](images/console.png)
      - Click on Service Accounts
        ![Service Account](images/service-account.png)
      - Create a new service account
        ![New Service Account](images/new-service-account.png)
      - Once the account is created, click on the service account and go to KEYS and click on ADD KEY
        ![Create Key](images/create-key.png)
      - Create a json key and once you click on create a credentials json file will be downloaded.
        ![JSON Key](images/json-key.png)
    - Enable all the required APIs - 
      ![Enabled APIs](images/enabled-apis.png)
        * Cloud Logging API
        * Cloud Monitoring API
        * Cloud Dataproc API
        * Cloud Dataproc Control API
        * Compute Engine API
        * Vertex AI API
        * BigQuery API
        * BigQuery Migration API
        * BigQuery Storage API
        * Cloud Datastore API
        * Cloud OS Login API
        * Cloud SQL
        * Cloud Storage
        * Cloud Storage API
        * Cloud Trace API
        * Google Cloud API
        * Google Cloud Storage JSON API
        * Service Management API
        * Service Usage API
    - Set up a version control repository for code and documentation.
      - Follow the steps to create new repository: https://docs.github.com/en/get-started/quickstart/create-a-repo
      - Clone the repository to your local

        ```sh
        git clone <repo-name>
        ```

2. **Infrastructure as Code with Terraform:**
   - Prerequisites:
     - Install Terraform: https://developer.hashicorp.com/terraform/downloads
   - Steps:
      - Create folder with name terraform.
      - Add service-account creds file and rename to `gcp-creds.json`.
      - Add train.csv file to data folder and preprocessing script to scripts folder.
      - Create a `main.tf` file with resource definitions (Feature Store, Dataproc, Bigquery).
      - Deploy the infra by applying terraform files.

      ```sh
      terraform init
      terraform apply -auto-approve 
      ```

      >NOTE:The file paths needs to be changed based on operating system, current paths are based on MAC environment, for Windows change accordingly.
