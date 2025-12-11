import pickle
from pathlib import Path
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

class _FeatureData:
    def __init__(self) -> None:
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self.mean_by_year: Dict[int, float] = {}
        self.std_by_year: Dict[int, float] = {}
    
    def is_set(self) -> bool:
        return (
            self.global_mean is not None
            and self.global_std is not None
            and len(self.mean_by_year) > 0
            and len(self.std_by_year) > 0
        )
    
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / "global_income_mean.pkl").open("wb") as f:
            pickle.dump(self.global_mean, f)
        with (directory / "global_income_std.pkl").open("wb") as f:
            pickle.dump(self.global_std, f)
        with (directory / "income_mean_by_year.pkl").open("wb") as f:
            pickle.dump(self.mean_by_year, f)
        with (directory / "income_std_by_year.pkl").open("wb") as f:
            pickle.dump(self.std_by_year, f)

    # note: load from file instead of mlflow
    @classmethod
    def load(cls, directory: str | Path) -> "_FeatureData":
        directory = Path(directory)

        obj = cls() 
        with (directory / "global_income_mean.pkl").open("rb") as f:
            obj.global_mean = pickle.load(f)
        with (directory / "global_income_std.pkl").open("rb") as f:
            obj.global_std = pickle.load(f)
        with (directory / "income_mean_by_year.pkl").open("rb") as f:
            obj.mean_by_year = pickle.load(f)
        with (directory / "income_std_by_year.pkl").open("rb") as f:
            obj.std_by_year = pickle.load(f)

        return obj

class FeatureExtractor:
    def __init__(self, state: Optional[_FeatureData] = None) -> None:
        self._state = state or _FeatureData()

    def get_features(self, data: DataFrame) -> DataFrame:
        if self._state.is_set():
            return self._get_features_inference(data)
        else:
            return self._get_features_training(data)
        
    # note: training and inference
    def _get_features_training(self, data: DataFrame) -> DataFrame:
        self._fit_state_from_data(data)
        return self._add_zscore_columns(data)
    
    def _get_features_inference(self, data: DataFrame) -> DataFrame:
        return self._add_zscore_columns(data)
    
    # note: fit z-score state from training data
    def _fit_state_from_data(self, data: DataFrame) -> None:
        global_stats = (
            data.select(
                F.mean(F.col("income")).alias("mean"),
                F.stddev_pop(F.col("income")).alias("std"),
            ).collect()[0]
        )

        global_mean = float(global_stats["mean"])
        global_std = float(global_stats["std"]) if global_stats["std"] is not None else 0.0
        if global_std == 0.0:
            global_std = 1.0 

        by_year_stats = (
            data.groupBy("year")
            .agg(
                F.mean(F.col("income")).alias("mean"),
                F.stddev_pop(F.col("income")).alias("std"),
            ).collect()
        )

        mean_by_year: Dict[int, float] = {}
        std_by_year: Dict[int, float] = {}
        for row in by_year_stats:
            year = int(row["year"])
            mean = float(row["mean"])
            std = float(row["std"]) if row["std"] is not None else 0.0
            if std == 0.0:
                std = 1.0
            mean_by_year[year] = mean
            std_by_year[year] = std

        self._state.global_mean = global_mean
        self._state.global_std = global_std
        self._state.mean_by_year = mean_by_year
        self._state.std_by_year = std_by_year

    def _add_zscore_columns(self, data: DataFrame) -> DataFrame:
        if not self._state.is_set():
            raise RuntimeError("Z-score state is not set, cannot compute z-scores.")

        data = data.withColumn("income_z", (F.col("income") - F.lit(self._state.global_mean)) / F.lit(self._state.global_std))

        spark = SparkSession.builder.getOrCreate()

        mean_rows = [(year, float(mean)) for year, mean in self._state.mean_by_year.items()]
        std_rows = [(year, float(std)) for year, std in self._state.std_by_year.items()]

        mean_df = spark.createDataFrame(mean_rows, schema=["year", "income_mean_year"])
        std_df = spark.createDataFrame(std_rows, schema=["year", "income_std_year"])

        data = data.join(mean_df, on="year", how="left")
        data = data.join(std_df, on="year", how="left")

        data = data.withColumn(
            "income_z_by_year",
            (F.col("income") - F.col("income_mean_year")) / F.col("income_std_year")
        )
        data = data.drop("income_mean_year", "income_std_year")

        return data
    
    def save_state(self, directory: str | Path) -> None:
        if not self._state.is_set():
            raise RuntimeError("Cannot save FeatureExtractor state: not fitted yet.")
        self._state.save(directory)

    @classmethod
    def load_state(cls, directory: str | Path) -> "FeatureExtractor":
        state = _FeatureData.load(directory)
        return cls(state=state)
