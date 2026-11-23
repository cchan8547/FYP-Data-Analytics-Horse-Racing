# Data Analytics for Horse Racing — FYP Project Handover

**Handover prepared:** 23 November 2026  
**Project:** Final Year Project / capstone  
**Main topic:** Hong Kong horse-racing probability prediction and WIN-betting evaluation  
**Current extension:** Timestamped WIN-odds collection and approximate horse-level money-flow reconstruction

## Read this first

This ZIP is a portable handover for a new FYP project. It contains the completed interim report, reproducible analysis source and data, an editable 9-minute presentation, and the standalone money-flow prototype. It is self-contained; access to the previous conversation is not required.

The empirical report and slides have already been completed and checked. The money-flow prototype is newer and has passed offline tests, but its live calculation has **not** yet been validated against an active HKJC market.

### Suggested first message in the new project

> Extract and inspect the attached FYP handover. Start with START_HERE.md. Treat the completed report and verified CSV results as the current baseline; do not silently refit models or change headline findings. Distinguish validated historical analysis from the provisional live money-flow extension. I will tell you which part to continue.

## Package map

| Path | Contents |
|---|---|
| `01_Report/` | Final 34-page interim report PDF. |
| `02_Analysis_Project/` | LaTeX source, collection and analysis scripts, raw and engineered data, result tables, and figures. |
| `03_Presentation/` | Editable 14-slide `.pptd` deck: ten timed main slides plus four Q&A backup slides, with page source and chart assets. |
| `04_Money_Flow_Prototype/` | Read-only public odds/pool collector, offline processing, synthetic example, tests, outputs, and its own detailed README. |
| `FILE_MANIFEST.csv` | File sizes and SHA-256 hashes for transfer verification. |

Large HTML caches, repeated PDF-page QA renders, temporary Python bytecode, and intermediate screenshots were deliberately excluded. No credentials, account details, or access tokens are included.

## 1. Current deliverables and status

### Interim report

- File: `01_Report/Horse_Racing_Capstone_Interim_Report.pdf`
- Language: English
- Length: 34 pages
- Report date: 19 November 2026
- Contains actual empirical analysis, not a proposed-work-only placeholder.
- LaTeX source: `02_Analysis_Project/report/main.tex` and the numbered section files.
- Bibliography: `02_Analysis_Project/report/references.bib`
- Student name, student ID, supervisor, department/programme, and institution remain bracketed placeholders because they were never supplied.

### Presentation

- Entry file: `03_Presentation/Horse_Racing_Capstone_9_Minute_Presentation.pptd`
- Ten main slides are timed to nine minutes, followed by four Q&A backup slides.
- The deck uses verified report charts and cautious interpretations.
- Slide source files are in `03_Presentation/pages/`; chart assets are in `03_Presentation/media/`.
- The included contact sheet is a visual preview, not the editable deliverable.
- No speaker notes were added.
- The deck previously passed validation with zero errors and zero warnings.
- The same identity placeholders remain editable.

### Money-flow prototype

- Folder: `04_Money_Flow_Prototype/`
- Main program: `hkjc_money_flow.py`
- Documentation: its local `README.md`
- Tests: 15 automated tests passed; standalone extraction and rerun also passed.
- Uses two read-only public GraphQL queries for WIN odds and WIN pool investment.
- Supports continuous collection, offline JSON/JSONL input, raw audit snapshots, normalized CSV output, interval estimates, restart recovery, and data-quality warnings.
- A real public request returned a future pre-sale pool with `STOP_SELL`, empty odds, and null investment. This validates the request path and graceful empty-state handling, **not** active-market money-flow accuracy.
- The included numerical demonstration is explicitly synthetic and must never be presented as observed HKJC money flow.

## 2. Verified research design

### Data provenance

The historical analysis uses official Hong Kong Jockey Club fixture and local-results pages. Collection metadata is in `02_Analysis_Project/data/collection_manifest.json`.

Collection snapshot:

- Retrieved: 19 November 2026
- Seasons: 2021/22 through 2025/26
- Meetings: 444
- Official races collected: 4,214
- Official result rows: 51,553
- Collection errors: 0
- Incomplete 2026/27 season excluded

Chronological design:

| Split | Seasons | Races | Runners |
|---|---|---:|---:|
| Training | 2021/22–2023/24 | 2,498 | 29,967 |
| Validation/calibration | 2024/25 | 847 | 10,341 |
| Untouched test | 2025/26 | 862 | 10,760 |

Final single-winner modeling sample:

- 4,207 races
- 51,068 starters
- 440 result meetings
- 404 withdrawn WV/WX rows removed before field-size and market calculations
- Seven dead-heat races excluded from the current single-winner likelihood

### Models

The main trained candidate is **Combined Conditional Logit**.

Each runner receives a linear score. Within-race softmax converts scores into probabilities summing to one per race. Training minimizes winner negative log likelihood plus ridge regularization.

- Fundamental models exclude all odds-derived information.
- Combined Logit adds `log_market_prob`.
- Combined LightGBM additionally uses `log_win_odds`, `market_rank`, and `overround`.
- Historical horse, jockey, trainer, course, and distance features are aggregated to entity–race level and chronologically shifted to avoid same-race leakage.
- Validation-selected ridge penalties: Fundamental Logit 0.1; Combined Logit 0.01.
- Temperature scaling is fitted on validation predictions only.

Evaluation includes race log loss, multiclass Brier loss, top-one strike rate, runner-level expected calibration error, runner-level ROC AUC with interpretation caveats, reliability diagrams, and paired meeting-cluster intervals.

## 3. Headline findings to preserve

### Prediction: test set

| Model | Log loss | Top-one strike rate |
|---|---:|---:|
| Uniform | 2.516217 | 11.9490% |
| Market | 2.032094 | 28.6543% |
| Fundamental Logit | 2.247077 | 22.8538% |
| Fundamental LightGBM | 2.246129 | 23.4339% |
| Combined Logit | **2.031766** | **29.3503%** |
| Combined LightGBM | 2.046935 | 28.5383% |

Additional Combined Logit test metrics: Brier 0.819465, ECE 0.005739, runner-level AUC 0.791983.

Validation Market log loss is 1.988639, better than every trained candidate. Combined Logit validation log loss is 2.001683, best among the four trained candidates.

Combined Logit minus Market paired test differences:

- Log loss: −0.000328; interval [−0.008867, +0.008288]
- Brier: +0.000650; interval [−0.002300, +0.003498]
- Top-one: +0.696 percentage points; interval [−0.692, +2.100] points

All three intervals include zero.

**Required interpretation:** Combined Logit is approximately market-comparable in this test sample; it is **not demonstrated to be superior to the market**. Do not describe it as beating the market.

### Betting design

The simulated value rule is:

- Expected value: `EV = p × decimal_odds − 1`
- EV threshold: greater than 0.05
- Decimal-odds cap: 30
- One highest-EV selection per qualifying race
- Flat HK$10 stake
- No rebate assumed
- Winners settled using official HKJC WIN dividends

Combined Logit value-strategy test results:

- 127 bets over 68 meetings
- HK$1,270 staked
- HK$91 profit
- ROI +7.17%
- 18 winners; strike rate 14.17%
- Mean odds 9.51
- Maximum drawdown HK$287
- 95% ROI interval [−49.56%, +77.19%]

Other test ROIs:

- Combined Logit top pick: −16.23%
- Combined LightGBM value: −2.12%
- Fundamental Logit value: −31.85%
- Fundamental LightGBM value: −14.32%
- Favourite every race: −19.94%
- Every runner: −23.55%

Threshold sensitivity is non-monotonic. For example, 0% edge/cap 30 gives −8.7% ROI over 269 bets; 5% edge/cap 30 gives +7.2% over 127 bets; 10% edge/cap 30 gives +65.9% over only 48 bets; 20% edge produces four losing bets; 30% edge produces no bets.

**Required interpretation:** the positive +7.17% ROI is small-sample and highly uncertain. It does not establish robust profitability. Test-grid threshold exploration is exploratory, not a prespecified out-of-sample betting rule.

## 4. Money-flow extension: exact current position

The working approximation uses local WIN-pool payout rate `r = 0.825`:

```text
implied_stake       = 0.825 × pool_investment / decimal_odds
implied_stake_share = implied_stake / pool_investment
implied_stake_flow  = current implied_stake − previous implied_stake
```

Synthetic check already passed:

- Pool HK$10,000,000 and decimal odds 5.0 imply HK$1,650,000.
- Later pool HK$11,000,000 and odds 4.5 imply approximately HK$2,016,666.67.
- Interval implied increase: approximately HK$366,666.67.

Interpretation constraints:

- These are market-implied allocations, not ticket-level transactions.
- They cannot identify bettors, bet counts, ticket sizes, or “smart money.”
- Odds/dividend rounding, refunds, withdrawals, stale snapshots, and asynchronous timestamps can distort estimates.
- Negative implied flow is retained and flagged; it is not automatically real money leaving a horse.
- Pool-wide growth can increase every runner's estimated stake. Stake-share change is therefore an essential companion metric.
- Any future predictive feature must use only information available at a fixed pre-race cutoff, with the market baseline evaluated at that same cutoff.
- Official final dividends remain suitable for after-race settlement, but do not prove earlier betting executability.

Live odds-field semantics remain unverified against an active market. Do not force the third-party decimal-odds convention onto archived or GraphQL fields without direct validation.

## 5. Highest-priority next work

1. **Resolve information timing and executability.** Final archived odds may not have been observable at the simulated betting time.
2. **Validate live odds semantics and synchronization.** On an active race day, compare GraphQL values, official pages, source timestamps, and subsequent official dividends.
3. **Add rolling-origin evaluation.** The current interim design has only one validation season and one test season.
4. **Run a small prespecified feature/model ablation.** Do not freely search the test set for a preferred result.
5. **Reassess betting only with chronologically frozen rules and genuinely new data.**
6. Add money-flow features only after the live collection and approximation have been validated; keep them separate from the current baseline until then.

Other acknowledged limitations: left-censored histories, simplified non-finish mechanisms, and incomplete sectional, equipment, trackwork, veterinary, weather, and related information.

## 6. Reproduction commands

Run commands from the extracted handover root unless a different directory is shown.

### Refit and regenerate analysis outputs

```bash
cd 02_Analysis_Project
python3 -m pip install -r requirements.txt
python3 scripts/analyze.py
python3 scripts/make_figures.py
```

`analyze.py` uses a fixed random seed (20261119) and reads/writes relative to its own project root. Regeneration overwrites existing derived outputs, so make a copy first if preserving the delivered baseline matters.

### Rebuild the report

```bash
cd 02_Analysis_Project/report
tectonic main.tex
```

The report was built with Tectonic. Before submission, replace identity placeholders and inspect the rebuilt PDF.

### Run the money-flow tests and synthetic demo

```bash
cd 04_Money_Flow_Prototype
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 hkjc_money_flow.py demo --output-dir my_demo
```

Use a new output directory for reruns because unchanged states are deliberately deduplicated.

### Perform one read-only live request

```bash
cd 04_Money_Flow_Prototype
python3 hkjc_money_flow.py live \
  --venue ST \
  --races 1 \
  --max-polls 1 \
  --output-dir live_smoke
```

For continuous race-day collection, use `--interval 60 --max-polls 0`; stop with Ctrl+C. No login or betting functionality exists.

## 7. Administrative checklist from the project instructions

- Interim report due: **26 November 2026 at 15:00**.
- Submit PDF or Word through the Final Year Project System.
- Filename pattern: `(Your Name) (Student ID)_Capstone Interim report.xxx`.
- Upload through Blackboard for Turnitin; stated acceptable similarity is 30%.
- Allow time for Turnitin generation.
- Submit a separate signed declaration form.
- Obtain supervisor review.
- Prepare the eventual 8–10 minute presentation followed by Q&A.

## 8. Working rules for the next assistant/session

- Read the actual report and result CSVs before making detailed claims.
- Preserve the distinction between training, validation, and untouched test data.
- Do not invent data, citations, student identity details, or live-market observations.
- Do not claim market superiority or a proven betting edge from the current results.
- Do not silently overwrite the report, analysis, result tables, or slide deck when extending the money-flow work.
- Keep raw observations separate from derived estimates.
- Keep synthetic fixtures visibly labeled as synthetic.
- The GitHub connection shown in the previous interface is not itself a transferred project workspace; files and authorization must still be handled explicitly in the new context.
