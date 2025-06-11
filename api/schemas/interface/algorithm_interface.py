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
