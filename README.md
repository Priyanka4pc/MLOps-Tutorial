# MLOps-Tutorial

## 1. Project Setup and Version Control:

- Create a new Google Cloud project.
  - Follow the steps to create a new project: https://console.cloud.google.com/projectcreate
  - Link a billing account to your google project
- Create a service account:
  - Go to https://console.cloud.google.com/
  - Click on IAM & Admin
    ![Console](images/console.png)
  - Click on Service Accounts
    ![Service Account](images/service-account.png)
  - Create a new service account and give owner permission (for tutorial purpose)
    ![New Service Account](images/new-service-account.png)
  - Once the account is created, click on the service account and go to KEYS and click on ADD KEY
    ![Create Key](images/create-key.png)
  - Create a json key and once you click on create a credentials json file will be downloaded.
    ![JSON Key](images/json-key.png)
- Enable all the required APIs - 
  ![Enabled APIs](images/enabled-apis.png)
  <!-- - Cloud Dataproc API -->
  - Cloud Resource Manager API
  - Vertex AI API
  - Secret Manager API
  - Cloud Build API
  - IAM Service Account Credentials API
- Set up a version control repository for code and documentation.
  - Follow the steps to create new repository: https://docs.github.com/en/get-started/quickstart/create-a-repo
  - Clone the repository to your local

    ```sh
    git clone <repo-name>
    ```

## 2. Infrastructure as Code with Terraform

- Prerequisites:
  - Install Terraform: https://developer.hashicorp.com/terraform/downloads
  - Create a bucket with name `tf-state-mlops` in GCP to store the terraform state.
    ![Tf State Bucket](ss-record/tf-state-bucket.gif)

- Steps:
  - Create folder with name terraform.
  - Add service-account creds file and rename to `gcp-creds.json`.
  - Add train.csv file to data folder and preprocessing script to scripts folder.
  - Create `provide.tf` to define google provider and set backed as gcs.
  - Create `main.tf` file with resource definitions (Feature Store, Dataproc, Bigquery).
  - Create `variable.tf` to store all the values to be passed by the user.
  - Create `output.tf` save all the outputs to be used later on.
  - Deploy the infra by applying terraform files. (Or will be taken care of using cloudbuild)

### Changes to be made

- Change bucket name in `provider.tf` to the bucket we just created.
- Variables to change in `variable.tf`:
  - `project_id`: change it to you google project ID
  - `creds_file`: change it to the path to your gcp credentials.json file
  - `suffix`: change it to some other value for uniqueness

>NOTE: The file paths needs to be changed based on operating system, current paths are based on MAC environment, for Windows change accordingly.

### Steps to run separately from cloudbuild

```sh
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp-creds.json"
terraform init
terraform apply -auto-approve 
```

## 3. Data Preprocessing pipeline

Define two components:

1. Preprocess - to preprocess the data and store to BQ
2. Feature store - to ingest the data from BQ to feature store

### Steps to run separately from cloudbuild

```sh
cd scripts
export PROJECT_ID=$PROJECT_ID # project id of the project
export REGION=$REGION # region used for this project
export DATASET=$DATASET # name of the bq dataset
export SRC_TABLE=$SRC_TABLE # name of the bq table
export DATA_BUCKET=$DATA_BUCKET # bucket that has training csv
export FEATURE_STORE_NAME=$FEATURE_STORE_NAME # feature store name
export ENTITY_TYPE_ID=$ENTITY_TYPE_ID # feature store entity name
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp-creds.json" # path to gcp credentials.json file

python preprocess-pipeline.py
```

## 4. Setup CloudBuild

- Enable Secret Manager API
- Go to: https://console.cloud.google.com/cloud-build/repositories/2nd-gen 
- ![Cloud Build](images/cloudbuild-repo.png)
- Click on create host connection and fill the details(connect to your github account)
  ![Create Host Connection](images/create-host-conn.png)
- Once done you should be able to see the connection
  ![Host](images/host.png)
- Now click on Link Repository and select the repo to link for ci/cd
  ![Link Repository](images/link-repo.png)
- Once done you should see ![Repo](images/repo.png)
- Now go to https://console.cloud.google.com/cloud-build/triggers
- Click on Create Trigger and fill all the required info
  ![Trigger Configuration](images/trigger-config1.png)
  ![Trigger Configuration](images/trigger-config2.png)
- Create a cloudbuild.yaml file in the github repo (steps mentioned in this file will be run in cloudbuild)
  - Change `SERVICE_ACCOUNT` value in line 34 and 47 to the service account we created
  - Change `GOOGLE_APPLICATION_CREDENTIALS` value in 11, 30 and 43 to the path to the creds file
- Now commit the changes to github, on every commit a cloudbuild pipeline will be triggered.
  ![Build](images/build.png)

## 5. Pipeline Creation

- Create a vertex pipeline with 3 steps
  - fetch features from feature store
  - train model
  - deploy model
  - model monitoring
- In cloudbuild.yaml change the service account name
- Once the cloudbuild pipeline runs, the vertex pipeline will be triggered
 ![Pipeline](images/pipeline.png)
- Wait for the pipeline to complete. Once the pipeline completes, you can see the model deployed on the endpoint which can be used for inferencing.
 ![Model Endpoint](images/model-endpoint.png)

### Steps to run separately from cloudbuild

 ```sh
cd scripts
export PROJECT_ID=$PROJECT_ID # project id of the project
export REGION=$REGION # region used for this project
export DATASET=$DATASET # name of the bq dataset
export SRC_TABLE=$SRC_TABLE # name of the bq table
export FEATURE_STORE_NAME=$FEATURE_STORE_NAME # feature store name
export ENTITY_TYPE_ID=$ENTITY_TYPE_ID # feature store entity name
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp-creds.json" # path to gcp credentials.json file

python pipeline.py
```

## 6. Model Monitoring and retraining

1. Once the monitoring pipeline starts running, add the notification channel for alert configuration
2. Create cloud function and add pub/sub trigger and upload the retraining pipeline to run whenever an alert comes.

>> NOTE: 1000 prediction requests are required for monitoring pipeline to start and once the pipeline is in running state, to send alert to notification channel setting needs to be configured manually once the monitoring pipeline is running.
