import pandas as pd
from configs.configs import PathsConfig, RunConfig

from src.features import *
from src.model_training import Model

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # note: #1: drop 'address' column as it is not useful for modeling
    data = data.drop(columns=["address"])

    ransom_df = data[data["label"] != "white"]
    white_df = data[data["label"] == "white"].sample(
        n=len(ransom_df), random_state=42
    )

    data = pd.concat([ransom_df, white_df]).sample(frac=1, random_state=RunConfig.random_seed)
    
    # test idea
    assert len(data[data['label']=='white']) == len(data[data['label']!='white'])

    # note: #2: create binary target variable 'is_ransomware'
    data['is_ransomware'] = (data['label'] != 'white').astype(int)
    data = data.drop(columns=['label'])

    return data

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = get_log_transformed_features(data, ['income', 'weight', 'count', 'looped'])
    data = get_ratio_features(data)
    data = get_temporal_features(data)
    #TODO: implement tests
    drop_cols = ["year", "income", "length"]
    data = data.drop(columns=drop_cols)

    print(f"Final feature set columns: {data.columns.tolist()}")
    return data

def model_training(data: pd.DataFrame) -> None:
    model = Model()
    print("\nCross-validation scores:")
    print(model.get_cv_scores(data))
    model.train_model(data)

    if RunConfig.evaluate_model:
        print("\nEvaluating model on test set:")
        model.evaluate_model(data)
    model.save_model()

def main():
    data = pd.read_csv(PathsConfig.raw_data)
    data = data_preprocessing(data)
    data = feature_engineering(data)
    model_training(data)

if __name__ == "__main__":
    main()