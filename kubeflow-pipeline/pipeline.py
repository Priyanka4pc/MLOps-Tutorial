from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component, ParallelFor, Input, Output, Artifact
from google.cloud import aiplatform

@component(packages_to_install=["scikit-learn", "xgboost", "pandas", "joblib"])
def train_model_op(model_name: str, model: Output[Artifact], model_score: Output[float]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import joblib

    data = pd.read_csv("/Users/prichoud3/Documents/MLOps-Tutorial/scripts/preprocessed.csv")
    X = data.drop("Price")
    Y = data["Price"]

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=0)

    if model_name=="LinearRegression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_name=="RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(max_leaf_nodes=100, random_state=1)
    elif model_name=="XGBRegressor":
        from xgboost import XGBRegressor
        model = XGBRegressor()

    model.fit(train_X, train_Y)
    model_path = f'{model_name}.joblib'
    joblib.dump(model, model_path)
    model.uri = model_path

    model_score = model.score(test_X, test_Y)

@component(packages_to_install=["scikit-learn"])
def select_best_model_op(models: list, accuracies: list, best_model: Output[Artifact]):
    # Find the index of the model with the highest accuracy
    best_index = accuracies.index(max(accuracies))

    # Select the best model
    best_model.uri = models[best_index].uri

def deploy_model_op(model: Input[Artifact]):
    # Initialize the Vertex AI Python SDK client
    client = aiplatform.gapic.ModelServiceClient()

    # Specify the project and location
    project = 'my-project' # replace with your project ID
    location = 'us-central1' # replace with your location

    # Specify the model and endpoint details
    model_display_name = 'my-model' # replace with your model display name
    endpoint_display_name = 'my-endpoint' # replace with your endpoint display name

    # Upload the model
    model = client.upload_model(
        parent=f'projects/{project}/locations/{location}',
        model=aiplatform.gapic.Model(display_name=model_display_name, artifact_uri=model.uri)
    )

    # Create the endpoint
    endpoint = client.create_endpoint(
        parent=f'projects/{project}/locations/{location}',
        endpoint=aiplatform.gapic.Endpoint(display_name=endpoint_display_name)
    )

    # Deploy the model to the endpoint
    client.deploy_model(
        endpoint=endpoint.name,
        deployed_model=aiplatform.gapic.DeployedModel(model=model.name),
        traffic_split={'0': 100}
    )

@dsl.pipeline(
    name='Train multiple models',
    description='A pipeline that trains multiple models in parallel and selects best one to deploy'
)
def train_multiple_models_pipeline():
    model_names = ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor'] # replace with your model names

    train_ops = []
    with dsl.ParallelFor(model_names) as model_name:
        train_ops.append(train_model_op(model_name))

    best_model_op = select_best_model_op([train_op.outputs['model'] for train_op in train_ops],
             [train_op.outputs['accuracy'] for train_op in train_ops])

    deploy_model_op(best_model_op.outputs['best_model'])

if __name__ == '__main__':
    compiler.Compiler().compile(train_multiple_models_pipeline, 'train_multiple_models_pipeline.json')