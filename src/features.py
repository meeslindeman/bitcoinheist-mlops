import pandas as pd
import numpy as np

def get_log_transformed_features(data: pd.DataFrame, features: list) -> pd.DataFrame:
    for feature in features:
        log_feature = f'log_{feature}'
        data[log_feature] = np.log1p(data[feature])
    return data

def get_ratio_features(data: pd.DataFrame) -> pd.DataFrame:
    data["activity_index"] = (data["count"] + data["neighbors"]) / (data["looped"] + 1)
    data["weight_to_count"] = data["weight"] / (data["count"] + 1)
    data["income_per_neighbor"] = data["income"] / (data["neighbors"] + 1)
    return data

def get_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    data["days_since_2009"] = (data["year"] - 2009) * 365 + data["day"]

    data["sin_day"] = np.sin(2 * np.pi * data["day"] / 365)
    data["cos_day"] = np.cos(2 * np.pi * data["day"] / 365)
    return data