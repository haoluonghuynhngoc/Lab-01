"""
FastAPI application for Movie Rating Prediction.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, MODEL_VERSION
from app.model import MovieRatingModel
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RecommendationResponse,
    RecommendationItem,
    UserInfoResponse,
    MovieInfoResponse,
    ModelInfoResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: MovieRatingModel = None


@app.on_event("startup")
async def startup_event():
    """Load model when application starts."""
    global model
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def _require_model() -> MovieRatingModel:
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict movie rating for a user.

    Try with sample IDs from MovieLens 100K, e.g. user_id=`196`, movie_id=`242`.
    """
    mdl = _require_model()

    try:
        rating = mdl.predict(request.user_id, request.movie_id)
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=rating,
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict movie ratings for multiple user-movie pairs in one request.
    """
    mdl = _require_model()

    try:
        results = []
        for item in request.predictions:
            rating = mdl.predict(item.user_id, item.movie_id)
            results.append(
                PredictionResponse(
                    user_id=item.user_id,
                    movie_id=item.movie_id,
                    predicted_rating=rating,
                    model_version=MODEL_VERSION,
                )
            )
        return BatchPredictionResponse(predictions=results, total_count=len(results))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendation"],
)
async def recommendations(
    user_id: str,
    top_n: int = Query(5, ge=1, le=50, description="Number of movies to return"),
):
    """
    Return top-N movie recommendations for a user (highest predicted ratings).

    Example: `/recommendations/196?top_n=5`
    """
    mdl = _require_model()

    try:
        recs = mdl.recommend(user_id, top_n=top_n)
        return RecommendationResponse(
            user_id=user_id,
            top_n=top_n,
            recommendations=[RecommendationItem(**r) for r in recs],
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}", response_model=UserInfoResponse, tags=["Lookup"])
async def get_user_info(user_id: str):
    """
    Check whether a user exists in the training set and list sample rated movies.
    """
    mdl = _require_model()
    rated = mdl.get_user_rated_movies(user_id)
    return UserInfoResponse(
        user_id=user_id,
        known_user=mdl.knows_user(user_id),
        n_rated_movies=len(rated),
        sample_rated_movies=rated[:10],
    )


@app.get("/movies/{movie_id}", response_model=MovieInfoResponse, tags=["Lookup"])
async def get_movie_info(movie_id: str):
    """
    Check whether a movie exists in the training set.
    """
    mdl = _require_model()
    return MovieInfoResponse(
        movie_id=movie_id,
        known_movie=mdl.knows_movie(movie_id),
    )


@app.get("/demo/samples", tags=["Demo"])
async def demo_samples():
    """
    Ready-made sample payloads for instructors / testers.

    Use these IDs directly in Swagger or curl.
    """
    return {
        "dataset": "MovieLens 100K",
        "note": "Copy any sample below into /predict or /predict/batch",
        "single_predict": {
            "user_id": "196",
            "movie_id": "242",
        },
        "batch_predict": {
            "predictions": [
                {"user_id": "196", "movie_id": "242"},
                {"user_id": "196", "movie_id": "302"},
                {"user_id": "22", "movie_id": "377"},
                {"user_id": "298", "movie_id": "474"},
            ]
        },
        "recommendation_examples": [
            "/recommendations/196?top_n=5",
            "/recommendations/22?top_n=10",
        ],
        "lookup_examples": [
            "/users/196",
            "/movies/242",
        ],
        "known_user_ids": ["196", "22", "298", "6", "1"],
        "known_movie_ids": ["242", "302", "377", "474", "50", "181"],
    }


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information and quick links."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "demo_samples": "/demo/samples",
        "endpoints": [
            "GET /health",
            "POST /predict",
            "POST /predict/batch",
            "GET /recommendations/{user_id}",
            "GET /users/{user_id}",
            "GET /movies/{movie_id}",
            "GET /model/info",
            "GET /demo/samples",
        ],
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get information about the loaded model, including training-set stats."""
    base = {
        "model_version": MODEL_VERSION,
        "model_type": "SVD (Collaborative Filtering)",
        "is_loaded": model is not None and model.is_loaded(),
    }
    if model is not None and model.is_loaded():
        base.update(model.get_stats())
    return ModelInfoResponse(**base)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
