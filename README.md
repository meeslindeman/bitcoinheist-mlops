# BitcoinHeist ML Pipeline (Course Project)

A compact reference for running and developing the BitcoinHeist end-to-end ML pipeline.  
Features: reproducible PySpark preprocessing, MySQL-backed storage, Dockerized execution, and classical ML modeling.

---

## References

- Dataset (UCI): https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset  
- Paper (feature definitions and analysis): https://arxiv.org/pdf/1906.07852  
- External GitHub implementation: https://github.com/toji-ut/BitcoinHeistRansomwareAnalytics  

Set project path manually (if needed):
```bash
export PYTHONPATH=$(pwd)
```

## Dataset Feature Notes

| Feature | Definition | Usefullness |
| ------- |----------- | ----------- |
| **Income** | Total coins received (`sum(outputs → address)`) | Captures payment magnitude — ransom payments cluster around specific BTC amounts. |
| **Neighbors** | Number of transactions sending to that address | Ransomware wallets often have few unique payers, unlike exchanges with many. |
| **Weight** | Sum of fractions of starter transactions’ coins reaching the address | Quantifies coin merging behavior (aggregation of payments). |
| **Length** | Longest chain length from a “starter” transaction to the address | Indicates how deep in the transaction graph the address sits (useful for detecting coin-mixing). |
| **Count** | Number of distinct starter transactions connected through chains | Measures how many separate flows converge to that address. |
| **Loop** | Number of starter transactions connected through *multiple* directed paths | Identifies obfuscation or coin-mixing loops. |

## Pipeline Overview
1. CSV → MySQL via `scripts/csv_to_mysql.py`.
2. Preprocessing in PySpark, saved to MySQL.
3. Feature engineering (log, ratios, temporal, z-scores), saved to MySQL.
4. Model training runs in Python on final feature table.

## Docker Commands
Build image:
```bash
docker build -f infra/build/Dockerfile -t bitcoinheist-app:latest .
```

Start full stack:
```bash
docker compose -f infra/docker-compose.yaml up
```

Force rebuild/restart:
```bash
docker compose -f infra/docker-compose.yaml up --force-recreate --remove-orphans
```

Shut down:
```bash
docker compose -f infra/docker-compose.yaml down
```

Rebuild image explicitly:
```bash
docker compose -f infra/docker-compose.yaml build --no-cache
```

## Notes

- The pipeline expects the BitcoinHeist CSV at `data/BitcoinHeistData.csv`.
- All intermediate data products are written to MySQL tables defined in `DatabaseConfig`.
- PySpark runs inside the application container using OpenJDK 17.
