"""
Unit tests for Movie Rating Prediction API.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create TestClient with startup/shutdown events."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client):
        """Test that health endpoint returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_format(self, client):
        """Test that health response has correct format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """Tests for the / endpoint."""

    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self, client):
        """Test that root response contains API information."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_input(self, client):
        """Test prediction with valid input."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rating" in data
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_response_format(self, client):
        """Test that prediction response has correct format."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"},
        )
        data = response.json()

        assert "user_id" in data
        assert "movie_id" in data
        assert "predicted_rating" in data
        assert "model_version" in data

    def test_predict_missing_user_id(self, client):
        """Test prediction with missing user_id."""
        response = client.post(
            "/predict",
            json={"movie_id": "242"},
        )
        assert response.status_code == 422

    def test_predict_missing_movie_id(self, client):
        """Test prediction with missing movie_id."""
        response = client.post(
            "/predict",
            json={"user_id": "196"},
        )
        assert response.status_code == 422

    def test_predict_empty_body(self, client):
        """Test prediction with empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestEdgeCases:
    """Edge case tests."""

    def test_predict_unknown_user(self, client):
        """Test prediction with unknown user ID."""
        response = client.post(
            "/predict",
            json={"user_id": "99999", "movie_id": "242"},
        )
        assert response.status_code == 200
        data = response.json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_unknown_movie(self, client):
        """Test prediction with unknown movie ID."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "99999"},
        )
        assert response.status_code == 200
        data = response.json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_special_characters_in_id(self, client):
        """Test prediction with special characters in IDs."""
        response = client.post(
            "/predict",
            json={"user_id": "user@#$", "movie_id": "movie!@#"},
        )
        # SVD still returns a default estimate for unknown IDs
        assert response.status_code == 200
        data = response.json()
        assert 1.0 <= data["predicted_rating"] <= 5.0


class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info_returns_200(self, client):
        """Test that model info endpoint returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_version(self, client):
        """Test that model info contains version."""
        response = client.get("/model/info")
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "is_loaded" in data
        assert data["is_loaded"] is True


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_multiple_items(self, client):
        """Test batch prediction with multiple items."""
        response = client.post(
            "/predict/batch",
            json={
                "predictions": [
                    {"user_id": "196", "movie_id": "242"},
                    {"user_id": "196", "movie_id": "302"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert 1.0 <= pred["predicted_rating"] <= 5.0

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"predictions": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["predictions"] == []


class TestRecommendationEndpoint:
    """Tests for the /recommendations endpoint."""

    def test_recommendations_returns_top_n(self, client):
        """Test top-N recommendations for a known user."""
        response = client.get("/recommendations/196?top_n=3")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "196"
        assert data["top_n"] == 3
        assert len(data["recommendations"]) == 3
        for item in data["recommendations"]:
            assert "movie_id" in item
            assert 1.0 <= item["predicted_rating"] <= 5.0


class TestLookupEndpoints:
    """Tests for user/movie lookup endpoints."""

    def test_user_info_known_user(self, client):
        response = client.get("/users/196")
        assert response.status_code == 200
        data = response.json()
        assert data["known_user"] is True
        assert data["n_rated_movies"] > 0

    def test_movie_info_known_movie(self, client):
        response = client.get("/movies/242")
        assert response.status_code == 200
        data = response.json()
        assert data["known_movie"] is True


class TestDemoSamplesEndpoint:
    """Tests for demo sample payloads."""

    def test_demo_samples(self, client):
        response = client.get("/demo/samples")
        assert response.status_code == 200
        data = response.json()
        assert "single_predict" in data
        assert "batch_predict" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
