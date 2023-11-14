import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from google.cloud import aiplatform

def preprocessing(data):

    data['CouncilArea'].fillna(data['CouncilArea'].mode()[0], inplace=True)
    data['YearBuilt'].fillna(data['YearBuilt'].mode()[0], inplace=True)
    data['BuildingArea'].fillna(data['BuildingArea'].mean(), inplace=True)

    # Convert "Date" column to date type
    data['time'] = pd.to_datetime(data['Date'])

    # Convert categorical columns to numerical using LabelEncoder
    categorical_features = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for column in categorical_features:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Drop columns
    columns_to_drop = ['Address', 'SellerG', 'Suburb', 'Date', 'Lattitude', 'Longtitude', 'Postcode']
    data.drop(columns_to_drop, axis=1, inplace=True)

    # Convert column names to lowercase
    data.columns = data.columns.str.lower()

    data = data.reset_index()
    data = data.rename(columns={"index":"index_column"})
    data["index_column"] = data["index_column"].astype(str)

    return data

data = pd.read_csv("../data/test_data_1.csv")
test_data = preprocessing(data.copy())

endpoint = aiplatform.Endpoint(endpoint_name=os.environ["ENDPOINT_ID"])
test_X = test_data.drop(['price', 'index_column', 'time'], axis=1)
resp = endpoint.predict(test_X.values.tolist())
df =  pd.DataFrame(data["Price"])
df["Predicted"]=resp.predictions
df["Predicted"] = df["Predicted"].round(2)
df.to_csv("preds.csv", index=False)