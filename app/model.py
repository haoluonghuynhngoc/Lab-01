"""
ML Model wrapper for movie rating prediction.
"""

import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any

from app.config import MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    Handles loading the trained model and making predictions.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise

    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)

        Returns:
            Predicted rating (float between 1.0 and 5.0)
        """
        prediction = self.model.predict(user_id, movie_id)
        return round(prediction.est, 2)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings
        """
        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def get_stats(self) -> Dict[str, Any]:
        """Return basic stats from the trained Surprise model."""
        trainset = self.model.trainset
        return {
            "n_users": trainset.n_users,
            "n_items": trainset.n_items,
            "n_ratings": trainset.n_ratings,
            "rating_scale": list(trainset.rating_scale),
            "global_mean": round(trainset.global_mean, 4),
        }

    def knows_user(self, user_id: str) -> bool:
        """Check if user exists in the training set."""
        try:
            self.model.trainset.to_inner_uid(user_id)
            return True
        except ValueError:
            return False

    def knows_movie(self, movie_id: str) -> bool:
        """Check if movie exists in the training set."""
        try:
            self.model.trainset.to_inner_iid(movie_id)
            return True
        except ValueError:
            return False

    def get_user_rated_movies(self, user_id: str) -> List[str]:
        """Return movie IDs the user rated during training."""
        trainset = self.model.trainset
        try:
            inner_uid = trainset.to_inner_uid(user_id)
        except ValueError:
            return []
        return [
            trainset.to_raw_iid(inner_iid)
            for inner_iid, _rating in trainset.ur[inner_uid]
        ]

    def recommend(self, user_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend top-N movies for a user (highest predicted ratings).

        Excludes movies the user already rated in the training set when possible.
        """
        trainset = self.model.trainset
        rated = set(self.get_user_rated_movies(user_id))
        candidates = [
            trainset.to_raw_iid(inner_iid)
            for inner_iid in trainset.all_items()
            if trainset.to_raw_iid(inner_iid) not in rated
        ]

        scored = [
            {"movie_id": movie_id, "predicted_rating": self.predict(user_id, movie_id)}
            for movie_id in candidates
        ]
        scored.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return scored[:top_n]


_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance
