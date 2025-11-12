from dataclasses import dataclass


@dataclass(init=False, frozen=True)
class PathsConfig:
    raw_data: str = "data/BitcoinHeistData.csv"  # Raw upstream data.
    model_path: str = "models/"  # Model store


@dataclass(init=False, frozen=True)
class RunConfig:
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2 


@dataclass(init=False, frozen=True)
class ModelConfig:
    model_name: str = "alpha"
    max_depth: int = 100
    class_weight: str = "balanced"