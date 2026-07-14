# Lab 1: Sản phẩm ML đầu tiên - Movie Rating Prediction API

## Tổng quan

API dự đoán rating phim dùng collaborative filtering (**SVD**), **FastAPI** và **Docker**.

Huấn luyện trên bộ dữ liệu **MovieLens 100K** (100.000 ratings, 943 users, 1.682 movies).

**Tính năng:**
- Dự đoán rating cho cặp user–movie
- Dự đoán theo lô (batch)
- Gợi ý top-N phim
- Tra cứu user / movie
- Payload mẫu sẵn để test nhanh
- Health check và thông tin model
- Triển khai Docker kèm health check
- Unit test với pytest

## Test nhanh cho giảng viên

1. Chạy API (xem phần bên dưới)
2. Mở Swagger: http://localhost:8000/docs
3. Hoặc xem payload mẫu: http://localhost:8000/demo/samples
4. Thử `POST /predict` với:

```json
{"user_id": "196", "movie_id": "242"}
```

## Cấu trúc dự án

```
ddm501-lab1-starter/
├── app/
│   ├── __init__.py
│   ├── main.py           # Ứng dụng FastAPI
│   ├── model.py          # Load model & dự đoán
│   ├── schemas.py        # Pydantic schemas
│   └── config.py         # Cấu hình
├── models/               # Model đã lưu (svd_model.pkl)
├── tests/
│   ├── __init__.py
│   └── test_api.py       # Unit tests
├── scripts/
│   └── train_model.py    # Script train model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Yêu cầu môi trường

- Python 3.10+
- Docker & Docker Compose
- Git

## Cài đặt

```bash
cd ddm501-lab1-starter

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate
# Kích hoạt (Linux/Mac)
# source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

## Huấn luyện model

```bash
python scripts/train_model.py
```

Script sẽ tải MovieLens 100K (qua thư viện Surprise) và lưu model SVD vào `models/svd_model.pkl`.

## Chạy API (local)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Liên kết | URL |
|----------|-----|
| Swagger UI | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Payload mẫu | http://localhost:8000/demo/samples |

## Danh sách API

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| GET | `/health` | Kiểm tra trạng thái API |
| POST | `/predict` | Dự đoán 1 rating |
| POST | `/predict/batch` | Dự đoán nhiều rating |
| GET | `/recommendations/{user_id}` | Gợi ý top-N phim |
| GET | `/users/{user_id}` | Tra cứu user + phim đã rate |
| GET | `/movies/{movie_id}` | Kiểm tra movie có trong tập train |
| GET | `/model/info` | Thông tin model + thống kê |
| GET | `/demo/samples` | Payload mẫu để test |
| GET | `/docs` | Tài liệu Swagger |

## Ví dụ sử dụng API

### 1. Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy", "model_loaded": true}
```

### 2. Dự đoán rating

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"196\", \"movie_id\": \"242\"}"
```

```json
{
  "user_id": "196",
  "movie_id": "242",
  "predicted_rating": 3.87,
  "model_version": "1.0.0"
}
```

### 3. Dự đoán batch

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d "{\"predictions\": [{\"user_id\": \"196\", \"movie_id\": \"242\"}, {\"user_id\": \"196\", \"movie_id\": \"302\"}, {\"user_id\": \"22\", \"movie_id\": \"377\"}]}"
```

### 4. Gợi ý phim (top 5)

```bash
curl "http://localhost:8000/recommendations/196?top_n=5"
```

### 5. Tra cứu user / movie

```bash
curl http://localhost:8000/users/196
curl http://localhost:8000/movies/242
```

### 6. Thông tin model

```bash
curl http://localhost:8000/model/info
```

### 7. Payload mẫu (cho người chấm bài)

```bash
curl http://localhost:8000/demo/samples
```

## ID gợi ý để test (MovieLens 100K)

| user_id | movie_id |
|---------|----------|
| 196 | 242 |
| 196 | 302 |
| 22 | 377 |
| 298 | 474 |
| 1 | 50 |

## Chạy kiểm thử

```bash
# Chạy toàn bộ test
pytest tests/ -v

# Chạy kèm coverage
pytest tests/ -v --cov=app --cov-report=html
```

## Chạy bằng Docker

Đảm bảo **Docker Desktop** đang chạy, sau đó:

```bash
docker-compose build
docker-compose up -d
```

API truy cập tại http://localhost:8000

Dừng dịch vụ:

```bash
docker-compose down
```

## Rubric chấm điểm

| Tiêu chí | Trọng số |
|----------|----------|
| Working ML Model | 25% |
| REST API | 25% |
| Docker Setup | 20% |
| Test Cases | 20% |
| Documentation | 10% |

## Nộp bài

Đẩy repository lên GitHub và nộp link repository.
