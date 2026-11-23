# HKJC WIN 賠率與估算資金流原型

這是一個**唯讀、可離線測試**的研究程式：定時讀取公開 GraphQL 的 WIN 賠率與 WIN 彩池 investment，保存原始快照，再估算每匹馬的累計注額、注額份額及相鄰快照之間的變化。

程式沒有登入、投注、帳戶存取或自動下注功能，也不會繞過 API 的查詢限制。

## 重要界定

- 輸出是 **market-implied stake allocation（市場隱含注額）**，不是真實注單交易紀錄。
- 無法由此識別投注者、注單數目、每張注單金額或所謂「精明錢」。
- 預設派彩率 `0.825` 是本地 WIN 彩池近似公式的可設定假設；使用前應核對研究期間適用的官方規則。
- 即時 `oddsValue` 的實際賠率口徑仍須在開售市場中，與官方頁面及最終派彩對照。非模擬數據預設帶有 `ODDS_CONVENTION_UNVERIFIED` 警告。
- 目前只完成公開查詢及售前空白回應的實測；**尚未取得活躍賽日的真實資金流結果**。

## 1. 快速開始

需要 Python 3.10 或以上；離線計算只使用標準庫，即時收集需要 `requests`。

```bash
cd HKJC_Money_Flow_Prototype
python3 -m pip install -r requirements.txt
```

執行明確標示為模擬數據的完整示範：

```bash
python3 hkjc_money_flow.py demo --output-dir my_demo
```

執行自動測試：

```bash
python3 -m unittest discover -s tests -v
```

壓縮包內的 `example_output/` 是已運行的示範結果。若要重新運行，請使用新的輸出目錄，例如 `my_demo`，避免既有狀態被判為重複。

## 2. 賽日即時收集

先對第 1 場做一次唯讀測試：

```bash
python3 hkjc_money_flow.py live \
  --venue ST \
  --races 1 \
  --max-polls 1 \
  --output-dir live_smoke
```

持續收集多場比賽，每 60 秒一輪：

```bash
python3 hkjc_money_flow.py live \
  --venue ST \
  --races 1 2 3 4 5 6 7 8 9 10 11 \
  --interval 60 \
  --max-polls 0 \
  --output-dir live_data
```

`--max-polls 0` 表示持續運行，按 `Ctrl+C` 安全停止；已寫入的 JSONL 與 CSV 會保留。沙田用 `ST`，跑馬地用 `HV`。程式會順序發出賠率與彩池查詢，預設逾時 15 秒、重試 3 次。

可選 `--date YYYY-MM-DD` 會放入請求，但這條公開路由主要面向當前或即將舉行的賽事，**不應視為歷史資料查詢接口**。`venue` 是請求標籤；正式研究時應用 pool ID、官方賽期及排位表再次核對。

請保持保守輪詢間隔，並遵守資料提供方適用的使用條款。公開接口的欄位或可用性可能改變。

## 3. 計算方法

設：

- `T_t`：時間 `t` 的 WIN 彩池 investment；
- `d_i,t`：馬匹 `i` 的十進制賠率，包括本金；
- `r`：派彩率，預設 `0.825`；
- `B_i,t`：未知的累計注額。

近似關係：

```text
d_i,t ≈ r × T_t / B_i,t
```

因此：

```text
implied_stake       = r × T_t / d_i,t
implied_stake_share = implied_stake / T_t
implied_stake_flow  = 本快照 implied_stake − 上一有效快照 implied_stake
```

已驗證的模擬例子：

```text
T = HK$10,000,000，d = 5.0
implied_stake = 0.825 × 10,000,000 / 5.0 = HK$1,650,000

之後 T = HK$11,000,000，d = 4.5
implied_stake ≈ HK$2,016,666.67
implied_stake_flow ≈ HK$366,666.67
```

**解讀資金流時應同時看份額變化。** 如果總彩池增加而賠率不變，估算累計注額會上升，但 `implied_stake_share_change = 0`；這不等於該馬相對其他馬獲得額外支持。

彩池閉合診斷：

```text
pool_closure_error    = Σ implied_stake − T
pool_closure_relative = pool_closure_error / T
```

在無捨入、無退款及公式完全成立時，二者應接近 0。現實中它們主要用來衡量近似是否可靠，而不是校準至必然為 0。

## 4. 輸出檔案

每次運行會在指定輸出目錄建立：

| 檔案 | 用途 |
|---|---|
| `raw_snapshots.jsonl` | 附加式保存每筆原始輸入或 GraphQL 回應，包括請求時間、回應時間及錯誤。 |
| `race_snapshots.csv` | 每場每個快照一列：彩池、來源時間、同步差距、閉合誤差及快照級警告。 |
| `runner_money_flow.csv` | 每匹馬的原始賠率、估算注額、份額、區間變化及馬匹級警告。 |
| `collection_events.jsonl` | 每筆處理結果摘要，方便稽核無數據、重複及錯誤狀態。 |

原始快照永遠保留。若彩池與賠率構成的衍生狀態沒有改變，`race_snapshots.csv` 會標記 `duplicate_unchanged_state`，但不會再新增相同的馬匹資金流行。程式重新啟動時，會從既有 CSV 載入上一個有效狀態，繼續計算區間變化。

主要馬匹欄位：

- `raw_odds_value`：未修改的 `oddsValue`；
- `decimal_odds`：經 `odds_divisor` 處理後、實際用於公式的賠率；
- `implied_stake`：估算累計注額，港元；
- `implied_stake_share`：估算注額佔總彩池比例；
- `implied_stake_flow`：相對上一有效快照的估算注額變化；
- `implied_stake_share_change`：相對上一有效快照的份額變化；
- `implied_flow_per_minute`：按觀察時間間距換算的近似變化率；
- `flow_status`：首個有效快照、已估算、負流量、輸入不足或不可比較等原因；
- `quality_flags`：以 `|` 分隔的警告。

## 5. 數據品質警告

| 警告 | 意思與處理 |
|---|---|
| `ODDS_CONVENTION_UNVERIFIED` | 非模擬輸入的賠率口徑尚未用真實開售市場核實；估算只屬暫定性。 |
| `EMPTY_ODDS_NODES` / `NO_WIN_POOL` | 沒有可計算的賠率或 WIN pool；不生成馬匹估算。 |
| `NOT_SELLING` | 回應顯示 `STOP_SELL`，常見於售前或停止投注後。 |
| `MISSING_INVESTMENT` / `NONPOSITIVE_INVESTMENT` | 保留賠率，但不計注額。 |
| `INVALID_ODDS` | 保留原始值，例如非數字或零；該馬不計注額。 |
| `MISSING_*_SOURCE_TIME` | 無法直接核實賠率與彩池是否同步，只保留觀察時間作稽核。 |
| `SOURCE_TIMESTAMP_MISMATCH` | 兩個來源時間差距超過 `--max-source-gap-seconds`，預設 120 秒；仍保留估算但明確警告。 |
| `STALE_*_SOURCE` | 來源時間距觀察時間過久，預設門檻 300 秒；也可能只是市場長時間未變，需人工判斷。 |
| `POOL_ID_MISMATCH` | 賠率與 investment 回應指向不同 pool，不應未經檢查直接合併。 |
| `NEGATIVE_IMPLIED_FLOW` | 保留負數並警告；可能來自賠率捨入、退款、退出馬匹、時間錯配或公式誤差，不代表真實提款。 |
| `POOL_CLOSURE_LARGE` | 全部可計算馬匹的隱含注額與總彩池相差較大，預設相對誤差門檻 20%。 |
| `RUNNERS_MISSING` / `RUNNERS_ADDED` | 相鄰快照的馬匹組合改變，需核查退出、復出或缺數。 |
| `NON_CHRONOLOGICAL_SNAPSHOT` | 快照時間倒退，不計該區間流量。 |
| `UNCHANGED_STATE` | 衍生輸入未變，原始回應保留，但不重複輸出馬匹估算。 |
| `SYNTHETIC_DATA` | 模擬輸入，絕不可當成真實 HKJC 觀察。 |

來源時間若是沒有時區的字串，程式按香港時間（UTC+8）解析，再另存 UTC。`lastUpdateTime` 的確切伺服器語義仍需在活躍市場中驗證；較舊時間不一定等於收集失敗。

## 6. 離線 JSON 輸入

可處理一個 JSON 陣列、含 `snapshots` 的 JSON 物件，或每行一個物件的 JSONL。最小格式如下：

```json
{
  "is_synthetic": true,
  "snapshots": [
    {
      "observed_at_utc": "2026-01-05T02:00:00Z",
      "meeting_date": "2026-01-05",
      "venue": "ST",
      "race_no": 1,
      "pool_id": "SYNTH-R1-WIN",
      "status": "OPEN",
      "sell_status": "SELLING",
      "odds_source_time": "2026-01-05T02:00:00Z",
      "pool_source_time": "2026-01-05T02:00:00Z",
      "pool_investment": 10000000,
      "odds": [
        {"combString": "1", "oddsValue": 5.0}
      ]
    }
  ]
}
```

執行：

```bash
python3 hkjc_money_flow.py offline \
  --input examples/synthetic_snapshots.json \
  --output-dir offline_results
```

也可直接輸入程式保存的原始 GraphQL JSONL 記錄，重新生成規範化輸出：

```bash
python3 hkjc_money_flow.py offline \
  --input live_data/raw_snapshots.jsonl \
  --output-dir reprocessed_data
```

## 7. 研究使用前的必要驗證

1. 在活躍賽日同時保存公開頁面截圖、GraphQL 賠率、investment 與來源時間，確認 `oddsValue` 是否為「每 HK$1 包括本金的十進制回報」。必要時才調整 `--odds-divisor`，不能用換算掩蓋口徑差異。
2. 對照多個時間點的官方 WIN 頁面，檢查 `lastUpdateTime`、捨入、最低賠率及更新延遲。
3. 記錄退出馬匹、退款、合併彩池及停止銷售狀態；不要把負流量直接解讀為資金撤出。
4. 固定一個賽前截止時間，只用當時已可觀察的快照建立特徵；市場基準也必須使用同一截止時間。
5. 最終派彩可用於賽後結算，但不代表該派彩對應的賠率在較早截止時間已可投注。
6. 在完成上述驗證前，不要把這些估算加入現有預測模型，也不要用作投注依據。

## 8. 目錄

```text
hkjc_money_flow.py                 主程式、公開查詢、計算與輸出
examples/synthetic_snapshots.json  明確標示的模擬輸入
example_output/                    完整模擬輸出
validation/live_smoke/             真實公開查詢的售前空白回應，不是活躍市場
tests/test_money_flow.py           公式、異常輸入、去重與重新啟動測試
TEST_RESULTS.txt                   本次測試紀錄
requirements.txt                   即時收集依賴
```
