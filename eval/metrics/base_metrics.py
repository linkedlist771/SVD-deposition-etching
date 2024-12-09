from pydantic import BaseModel
from abc import ABC, abstractmethod
from pathlib import Path


class BaseMetrics(ABC, BaseModel):
    # Pydantic fields
    """
    Here, we require that all those dataset has the same structure.
    """
    real_dataset_dir_path: Path
    generated_dataset_dir_path: Path

    # Abstract method that must be implemented by subclasses
    @abstractmethod
    def calculate(self) -> float:
        raise NotImplementedError


#
# # Example subclass
# class PSNRMetric(BaseMetrics):
#     def calculate(self) -> float:
#         # Implement PSNR calculation
#         return self.value


