## Misc

https://arxiv.org/pdf/1906.07852
https://github.com/toji-ut/BitcoinHeistRansomwareAnalytics

`export PYTHONPATH=$(pwd)`

## Dataset

| Feature | Definition | Usefullness |
| ------- |----------- | ----------- |
| **Income** | Total coins received (`sum(outputs → address)`) | Captures payment magnitude — ransom payments cluster around specific BTC amounts. |
| **Neighbors** | Number of transactions sending to that address | Ransomware wallets often have few unique payers, unlike exchanges with many. |
| **Weight** | Sum of fractions of starter transactions’ coins reaching the address | Quantifies coin merging behavior (aggregation of payments). |
| **Length** | Longest chain length from a “starter” transaction to the address | Indicates how deep in the transaction graph the address sits (useful for detecting coin-mixing). |
| **Count** | Number of distinct starter transactions connected through chains | Measures how many separate flows converge to that address. |
| **Loop** | Number of starter transactions connected through *multiple* directed paths | Identifies obfuscation or coin-mixing loops. |
