from eval.metrics.base_metrics import BaseMetrics
from eval.utils.fid_utils import FIDConfig, calculate_fid


class FIDMetric(BaseMetrics):
    batch_size: int = 50
    dims: int = 2048

    def calculate(self) -> float:
        # Example usage:
        config = FIDConfig(
            batch_size=self.batch_size,
            # paths=("path/to/images1", "path/to/images2"),
            paths=(self.real_dataset_dir_path, self.generated_dataset_dir_path),
            dims=self.dims,
        )
        fid_value = calculate_fid(config)
        return float(fid_value)
