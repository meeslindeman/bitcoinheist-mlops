# note: normal call
curl \
  -H "Content-Type: application/json" \
  -d '{
        "year": 2014,
        "day": 150,
        "length": 5,
        "weight": 0.12,
        "count": 3,
        "looped": 1,
        "neighbors": 15,
        "income": 0.003
      }' \
  -X POST http://127.0.0.1:5001/predict


# note: negative / weird-value call
curl \
  -H "Content-Type: application/json" \
  -d '{
        "year": 2011,
        "day": -10,
        "length": -5,
        "weight": -0.12,
        "count": -3,
        "looped": -1,
        "neighbors": -10,
        "income": -50
      }' \
  -X POST http://127.0.0.1:5001/predict


# note: missing / unknown fields simulation
curl \
  -H "Content-Type: application/json" \
  -d '{
        "year": 2020,
        "day": 200,
        "income": 1.5
      }' \
  -X POST http://127.0.0.1:5001/predict
