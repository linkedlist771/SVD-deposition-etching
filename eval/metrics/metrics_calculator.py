from typing import List, Dict
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
from eval.metrics.base_metrics import BaseMetrics
from eval.metrics.fid import FIDMetric


class MetricsCalculator(BaseModel):
    real_dataset_dir_path: Path
    generated_dataset_dir_path: Path
    metrics: List[str]  # List of metric names to calculate

    # You might want to add more configuration parameters
    batch_size: int = 50 # this is for the fid
    dims: int = 2048

    def _get_metric_instance(self, metric_name: str) -> BaseMetrics:
        """
        Factory method to create metric instances based on metric names
        """
        metrics_mapping = {
            "fid": FIDMetric(
                real_dataset_dir_path=self.real_dataset_dir_path,
                generated_dataset_dir_path=self.generated_dataset_dir_path,
                batch_size=self.batch_size,
                dims=self.dims
            ),
            # Add more metrics here as needed
            # "psnr": PSNRMetric(
            #     real_dataset_dir_path=self.real_dataset_dir_path,
            #     generated_dataset_dir_path=self.generated_dataset_dir_path
            # ),
        }

        return metrics_mapping.get(metric_name.lower())

    def calculate_all(self) -> Dict[str, float]:
        """
        Calculate all specified metrics and return results in a dictionary
        """
        results = {}

        for metric_name in self.metrics:
            logger.info(f"Calculating metric '{metric_name}'")
            metric_instance = self._get_metric_instance(metric_name)
            if metric_instance is None:
                logger.warning(f"Warning: Metric '{metric_name}' not implemented")
                continue

            try:
                result = metric_instance.calculate()
                results[metric_name] = result
            except Exception as e:
                logger.error(f"Error calculating metric '{metric_name}': {str(e)}")
                results[metric_name] = None

        return results

if __name__ == "__main__":
    from configs.path_configs import ROOT

    calculator = MetricsCalculator(
        real_dataset_dir_path=Path(ROOT / "data/shaoji-data/12-05-01-03-19"),
        generated_dataset_dir_path=Path(ROOT / "data/shaoji-data/12-05-01-04-42"),
        metrics=["FID", "PSNR"]  # Add more metrics as needed
    )
    results = calculator.calculate_all()
    print(results)  # {'fid': 123.45, 'psnr': 67.89}