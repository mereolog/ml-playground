from typing import Type

from pydantic import BaseModel, Field


class AlgorithmInfo(BaseModel):
    internal_name: str = Field(
        default="",
        description="Internal name of the algorithm, used for internal identification.",
    )
    display_name: str = Field(
        default="",
        description="Display name of the algorithm, used for user interfaces.",
    )
    description: str = Field(
        default="",
        description="Description of the algorithm, providing details about its functionality.",
    )

class AlgorithmRegistryEntry:
    """Registry entry for ML algorithms containing metadata and configuration schema."""

    def __init__(self, info: AlgorithmInfo, pydantic_model: Type, algorithm_class: Type):
        self.info = info
        self.pydantic_model = pydantic_model
        self.algorithm_class = algorithm_class
