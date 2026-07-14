"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    user_id: str = Field(..., example="196", description="User ID")
    movie_id: str = Field(..., example="242", description="Movie ID")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    model_config = {"protected_namespaces": ()}

    user_id: str = Field(..., example="196")
    movie_id: str = Field(..., example="242")
    predicted_rating: float = Field(..., example=4.05, description="Predicted rating between 1.0 and 5.0")
    model_version: str = Field(..., example="1.0.0")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    model_config = {"protected_namespaces": ()}

    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)


class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""
    user_id: str = Field(..., example="196")
    movie_id: str = Field(..., example="242")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    predictions: List[PredictionItem] = Field(
        ...,
        example=[
            {"user_id": "196", "movie_id": "242"},
            {"user_id": "196", "movie_id": "302"},
        ],
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    predictions: List[PredictionResponse]
    total_count: int


class RecommendationItem(BaseModel):
    """Single recommendation result."""
    movie_id: str = Field(..., example="318")
    predicted_rating: float = Field(..., example=4.52)


class RecommendationResponse(BaseModel):
    """Top-N movie recommendations for a user."""
    model_config = {"protected_namespaces": ()}

    user_id: str
    top_n: int
    recommendations: List[RecommendationItem]
    model_version: str


class UserInfoResponse(BaseModel):
    """Information about a user in the training data."""
    user_id: str
    known_user: bool
    n_rated_movies: int
    sample_rated_movies: List[str]


class MovieInfoResponse(BaseModel):
    """Information about a movie in the training data."""
    movie_id: str
    known_movie: bool


class ModelInfoResponse(BaseModel):
    """Detailed model metadata."""
    model_config = {"protected_namespaces": ()}

    model_version: str
    model_type: str
    is_loaded: bool
    n_users: Optional[int] = None
    n_items: Optional[int] = None
    n_ratings: Optional[int] = None
    rating_scale: Optional[List[float]] = None
    global_mean: Optional[float] = None
