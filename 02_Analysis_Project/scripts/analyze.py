#!/usr/bin/env python3
"""Leakage-safe feature engineering, race-probability models, and WIN backtests."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 20261119
np.random.seed(SEED)


def add_entity_history(df, keys, prefix, alpha_win=30.0, alpha_place=30.0, alpha_finish=20.0):
    """Expanding per-mount history aggregated to race level before shifting."""
    cols = keys + ["race_id", "race_order", "won", "top3", "finish_pct"]
    temp = df[cols].copy()
    temp["mounts"] = 1
    agg = temp.groupby(keys + ["race_id"], as_index=False, sort=False).agg(
        race_order=("race_order", "first"), mounts=("mounts", "sum"), wins=("won", "sum"),
        places=("top3", "sum"), finish_sum=("finish_pct", "sum"))
    agg = agg.sort_values(keys + ["race_order"])
    g = agg.groupby(keys, sort=False)
    starts = g["mounts"].cumsum() - agg["mounts"]
    wins = g["wins"].cumsum() - agg["wins"]
    places = g["places"].cumsum() - agg["places"]
    finish_sum = g["finish_sum"].cumsum() - agg["finish_sum"]
    agg[prefix + "_starts"] = starts
    agg[prefix + "_win_rate"] = (wins + alpha_win / 12) / (starts + alpha_win)
    agg[prefix + "_place_rate"] = (places + alpha_place * 0.25) / (starts + alpha_place)
    agg[prefix + "_mean_finish"] = (finish_sum + alpha_finish * 0.5) / (starts + alpha_finish)
    keep = keys + ["race_id", prefix + "_starts", prefix + "_win_rate", prefix + "_place_rate", prefix + "_mean_finish"]
    return df.merge(agg[keep], on=keys + ["race_id"], how="left", sort=False)


def load_and_engineer():
    races = pd.read_csv(DATA / "races_raw.csv", parse_dates=["date"])
    runners = pd.read_csv(DATA / "runners_raw.csv", parse_dates=["date"])
    meta = races[["race_id", "class_text", "class_name", "distance_m", "rating_band", "going", "course_config", "purse"]]
    df = runners.merge(meta, on="race_id", how="left", validate="many_to_one")
    official_races = int(df["race_id"].nunique())
    official_runners = int(len(df))
    # HKJC result tables include withdrawn horses with WV/WX codes. They were not
    # starters, must not enter field sizes/market probabilities, and must not
    # cause the entire race to be discarded.
    nonstarter_codes = {"WV", "WV-A", "WX", "WX-A", "WXNR"}
    nonstarter = df["placing_raw"].astype(str).str.upper().isin(nonstarter_codes)
    unexpected_missing = df.loc[~nonstarter & (df["win_odds"].isna() | df["draw"].isna())]
    if not unexpected_missing.empty:
        raise ValueError(f"Unexpected starters with missing odds/draw: {len(unexpected_missing)}")
    withdrawn_runners = int(nonstarter.sum())
    withdrawn_races = int(df.loc[nonstarter, "race_id"].nunique())
    df = df.loc[~nonstarter].copy()
    df = df.sort_values(["date", "race_no", "course", "horse_no"]).reset_index(drop=True)
    df["race_order"] = df.groupby("race_id")["race_no"].transform("first")
    unique_races = df[["race_id", "date", "course", "race_no"]].drop_duplicates().sort_values(["date", "race_no", "course"]).reset_index(drop=True)
    unique_races["race_sequence"] = np.arange(len(unique_races))
    df = df.merge(unique_races[["race_id", "race_sequence"]], on="race_id", how="left")
    df["race_order"] = df["race_sequence"]
    df["field_size"] = df.groupby("race_id")["horse_no"].transform("size")
    # Extract numeric placings from dead-heat labels such as "1 DH" while
    # leaving pulled-up/unseated/non-finisher codes as missing finishes.
    df["placing_num"] = pd.to_numeric(df["placing_raw"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    df["finish_pct"] = (df["placing_num"] / df["field_size"]).fillna(1.0).clip(0, 1)
    df["top3"] = (df["placing_num"] <= 3).astype(int)
    df["won"] = (df["placing_num"] == 1).astype(int)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["surface"] = np.where(df["course_config"].fillna("").str.contains("ALL WEATHER", case=False), "AWT", "TURF")
    df["going"] = df["going"].fillna("UNKNOWN").replace("", "UNKNOWN")
    df["class_name"] = df["class_name"].fillna("UNKNOWN").replace("", "UNKNOWN")
    class_map = {"GRIFFIN": 0, "CLASS 5": 1, "CLASS 4": 2, "CLASS 3": 3, "CLASS 2": 4,
                 "CLASS 1": 5, "GROUP 3": 6, "GROUP 2": 7, "GROUP 1": 8}
    df["class_score"] = df["class_name"].str.upper().map(class_map).fillna(3.0)

    # Market probabilities are computed from the complete published field.
    df["raw_implied"] = 1 / df["win_odds"].where(df["win_odds"] >= 1)
    df["overround"] = df.groupby("race_id")["raw_implied"].transform("sum")
    df["market_prob"] = df["raw_implied"] / df["overround"]
    df["log_market_prob"] = np.log(df["market_prob"].clip(lower=1e-8))
    df["market_rank"] = df.groupby("race_id")["win_odds"].rank(method="min", ascending=True)
    df["log_win_odds"] = np.log(df["win_odds"].where(df["win_odds"] > 0))

    df = add_entity_history(df, ["horse_id"], "horse", 24, 24, 16)
    df = add_entity_history(df, ["horse_id", "distance_m"], "horse_dist", 12, 12, 8)
    df = add_entity_history(df, ["horse_id", "course"], "horse_course", 12, 12, 8)
    df = add_entity_history(df, ["jockey"], "jockey", 60, 60, 40)
    df = add_entity_history(df, ["trainer"], "trainer", 80, 80, 50)

    horse = df.groupby("horse_id", sort=False)
    df["horse_previous_date"] = horse["date"].shift()
    df["horse_days_since"] = (df["date"] - df["horse_previous_date"]).dt.days.fillna(365).clip(0, 365)
    df["horse_last_finish"] = horse["finish_pct"].shift().fillna(0.5)
    df["horse_last3_finish"] = horse["finish_pct"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean()).fillna(0.5)
    df["horse_last3_win"] = horse["won"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean()).fillna(1 / 12)
    for col in ["actual_weight_lbs", "horse_weight_lbs", "distance_m", "class_score"]:
        df["previous_" + col] = horse[col].shift()
    df["actual_weight_change"] = (df["actual_weight_lbs"] - df["previous_actual_weight_lbs"]).fillna(0)
    df["horse_weight_change"] = (df["horse_weight_lbs"] - df["previous_horse_weight_lbs"]).fillna(0)
    df["distance_change_km"] = ((df["distance_m"] - df["previous_distance_m"]) / 1000).fillna(0)
    df["class_change"] = (df["class_score"] - df["previous_class_score"]).fillna(0)

    # Race-relative and domain-interaction variables.
    df["actual_weight_rel"] = df["actual_weight_lbs"] - df.groupby("race_id")["actual_weight_lbs"].transform("mean")
    df["horse_weight_rel"] = df["horse_weight_lbs"] - df.groupby("race_id")["horse_weight_lbs"].transform("mean")
    df["draw_ratio"] = (df["draw"] - 1) / (df["field_size"] - 1).clip(lower=1)
    df["draw_ratio_sq"] = df["draw_ratio"] ** 2
    df["distance_km"] = df["distance_m"] / 1000
    df["course_hv"] = (df["course"] == "HV").astype(int)
    df["surface_awt"] = (df["surface"] == "AWT").astype(int)
    df["draw_x_sprint"] = df["draw_ratio"] * (df["distance_m"] <= 1200).astype(int)
    df["draw_x_hv"] = df["draw_ratio"] * df["course_hv"]
    df["draw_x_awt"] = df["draw_ratio"] * df["surface_awt"]
    for col in ["horse_starts", "horse_dist_starts", "horse_course_starts", "jockey_starts", "trainer_starts"]:
        df[col + "_log"] = np.log1p(df[col].fillna(0))
    df["horse_first_start"] = (df["horse_starts"].fillna(0) == 0).astype(int)
    df["horse_days_log"] = np.log1p(df["horse_days_since"])

    # A model race requires one winner, complete market prices and core fields, and an official WIN dividend.
    quality = df.groupby("race_id").agg(
        winners=("won", "sum"), field=("horse_no", "size"),
        odds_ok=("win_odds", lambda s: bool((s.notna() & (s >= 1)).all())),
        draw_ok=("draw", lambda s: bool(s.notna().all())),
        weight_ok=("actual_weight_lbs", lambda s: bool(s.notna().all())),
        distance_ok=("distance_m", lambda s: bool(s.notna().all())),
        winner_dividend=("win_dividend_hkd10", lambda s: float(s[df.loc[s.index, "won"] == 1].sum())))
    quality["valid"] = (quality["winners"] == 1) & quality["field"].between(4, 14) & quality["odds_ok"] & quality["draw_ok"] & quality["weight_ok"] & quality["distance_ok"] & (quality["winner_dividend"] > 0)
    valid_ids = set(quality.index[quality["valid"]])
    exclusions = {
        "official_races": official_races,
        "official_result_rows": official_runners,
        "withdrawn_rows_removed": withdrawn_runners,
        "races_with_withdrawals": withdrawn_races,
        "official_starters": official_runners - withdrawn_runners,
        "dead_heat_races": int((quality["winners"] > 1).sum()),
        "excluded_races": int((~quality["valid"]).sum()),
        "valid_races": int(quality["valid"].sum()),
        "exclusion_reason_counts": {
            "not_exactly_one_winner": int((quality["winners"] != 1).sum()),
            "field_outside_4_14": int((~quality["field"].between(4, 14)).sum()),
            "incomplete_odds": int((~quality["odds_ok"]).sum()),
            "incomplete_draw": int((~quality["draw_ok"]).sum()),
            "incomplete_actual_weight": int((~quality["weight_ok"]).sum()),
            "missing_distance": int((~quality["distance_ok"]).sum()),
            "missing_winner_dividend": int((quality["winner_dividend"] <= 0).sum()),
        },
    }
    model_df = df[df["race_id"].isin(valid_ids)].copy().sort_values(["race_order", "horse_no"]).reset_index(drop=True)
    return model_df, exclusions


LOGIT_FEATURES = [
    "actual_weight_rel", "horse_weight_rel", "draw_ratio", "draw_ratio_sq",
    "horse_starts_log", "horse_win_rate", "horse_place_rate", "horse_mean_finish",
    "horse_last_finish", "horse_last3_finish", "horse_last3_win", "horse_days_log", "horse_first_start",
    "horse_dist_starts_log", "horse_dist_win_rate", "horse_dist_place_rate", "horse_dist_mean_finish",
    "horse_course_starts_log", "horse_course_win_rate", "horse_course_place_rate", "horse_course_mean_finish",
    "jockey_starts_log", "jockey_win_rate", "jockey_place_rate", "jockey_mean_finish",
    "trainer_starts_log", "trainer_win_rate", "trainer_place_rate", "trainer_mean_finish",
    "actual_weight_change", "horse_weight_change", "distance_change_km", "class_change",
    "draw_x_sprint", "draw_x_hv", "draw_x_awt",
]
LGBM_NUMERIC = LOGIT_FEATURES + ["field_size", "distance_km", "class_score", "course_hv", "surface_awt"]
LGBM_CATEGORICAL = ["going", "class_name", "course_config", "course"]
MARKET_FEATURES = ["log_market_prob", "log_win_odds", "market_rank", "overround"]


def race_layout(frame):
    sizes = frame.groupby("race_id", sort=False).size().to_numpy()
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    winners = np.flatnonzero(frame["won"].to_numpy() == 1)
    return sizes, starts, winners


def normalise_by_race(frame, raw):
    raw = np.asarray(raw, dtype=float)
    raw = np.clip(raw, 1e-12, 1 - 1e-12)
    temp = pd.DataFrame({"race_id": frame["race_id"].to_numpy(), "raw": raw})
    denom = temp.groupby("race_id", sort=False)["raw"].transform("sum").to_numpy()
    return raw / denom


def race_metrics(frame, probs):
    sizes, starts, winners = race_layout(frame)
    probs = np.clip(np.asarray(probs), 1e-12, 1)
    log_loss = -np.log(probs[winners]).mean()
    squared = np.add.reduceat(probs ** 2, starts) - 2 * probs[winners] + 1
    top1 = 0
    for start, size, winner in zip(starts, sizes, winners):
        top1 += int(start + np.argmax(probs[start:start + size]) == winner)
    top1 /= len(sizes)
    y = frame["won"].to_numpy()
    ece_frame = pd.DataFrame({"p": probs, "y": y})
    try:
        ece_frame["bin"] = pd.qcut(ece_frame["p"], 10, duplicates="drop")
        grouped = ece_frame.groupby("bin", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
        ece = float((grouped["n"] / len(ece_frame) * (grouped["p"] - grouped["y"]).abs()).sum())
    except ValueError:
        ece = float("nan")
    return {"log_loss": float(log_loss), "brier": float(squared.mean()), "top1": float(top1),
            "ece": ece, "runner_auc": float(roc_auc_score(y, probs))}


def temperature_transform(frame, probs, temperature):
    probs = np.asarray(probs, dtype=float)
    scaled = np.log(np.clip(probs, 1e-12, 1)) / temperature
    sizes, starts, _ = race_layout(frame)
    maxima = np.maximum.reduceat(scaled, starts)
    expv = np.exp(scaled - np.repeat(maxima, sizes))
    denom = np.add.reduceat(expv, starts)
    return expv / np.repeat(denom, sizes)


def fit_temperature(frame, probs):
    objective = lambda log_t: -np.log(np.clip(temperature_transform(frame, probs, np.exp(log_t))[race_layout(frame)[2]], 1e-12, 1)).mean()
    result = minimize_scalar(objective, bounds=(-1.5, 1.5), method="bounded")
    return float(np.exp(result.x))


class ConditionalLogit:
    def __init__(self, penalty=0.001):
        self.penalty = penalty
        self.coef_ = None

    def fit(self, X, frame):
        X = np.asarray(X, dtype=np.float64)
        y = frame["won"].to_numpy(dtype=np.float64)
        sizes, starts, winners = race_layout(frame)
        n_races = len(sizes)

        def objective(w):
            score = X @ w
            maxima = np.maximum.reduceat(score, starts)
            expv = np.exp(np.clip(score - np.repeat(maxima, sizes), -700, 700))
            denom = np.add.reduceat(expv, starts)
            probs = expv / np.repeat(denom, sizes)
            loss = -np.log(np.clip(probs[winners], 1e-15, 1)).sum() / n_races + 0.5 * self.penalty * np.dot(w, w)
            grad = X.T @ (probs - y) / n_races + self.penalty * w
            return loss, grad

        result = minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7})
        if not result.success:
            print("Conditional logit warning:", result.message)
        self.coef_ = result.x
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.coef_

    def predict_proba(self, X, frame):
        score = self.decision_function(X)
        sizes, starts, _ = race_layout(frame)
        maxima = np.maximum.reduceat(score, starts)
        expv = np.exp(np.clip(score - np.repeat(maxima, sizes), -700, 700))
        return expv / np.repeat(np.add.reduceat(expv, starts), sizes)


def make_preprocessors(train, logit_features, lgbm_numeric):
    logit_prep = ColumnTransformer([("num", __import__("sklearn").pipeline.Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), logit_features)], remainder="drop")
    lgbm_prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), lgbm_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LGBM_CATEGORICAL),
    ], remainder="drop")
    logit_prep.fit(train)
    lgbm_prep.fit(train)
    return logit_prep, lgbm_prep


def train_models(train, validation):
    frames = {"train": train, "validation": validation}
    predictions = {split: {} for split in frames}
    artifacts = {"penalties": {}, "models": {}}
    for include_market in [False, True]:
        label = "combined" if include_market else "fundamental"
        logit_features = LOGIT_FEATURES + (["log_market_prob"] if include_market else [])
        lgbm_numeric = LGBM_NUMERIC + (MARKET_FEATURES if include_market else [])
        logit_prep, lgbm_prep = make_preprocessors(train, logit_features, lgbm_numeric)
        X_train = logit_prep.transform(train)
        X_val = logit_prep.transform(validation)
        best = None
        for penalty in [0.0001, 0.001, 0.01, 0.1]:
            model = ConditionalLogit(penalty).fit(X_train, train)
            p_val = model.predict_proba(X_val, validation)
            score = race_metrics(validation, p_val)["log_loss"]
            if best is None or score < best[0]:
                best = (score, penalty, model, p_val)
        _, penalty, logit_model, p_val = best
        artifacts["penalties"][label] = penalty
        predictions["validation"][label + "_logit"] = p_val
        predictions["train"][label + "_logit"] = logit_model.predict_proba(X_train, train)
        artifacts["models"][label + "_logit"] = (logit_model, logit_prep, logit_features)

        X_train_l = lgbm_prep.transform(train)
        X_val_l = lgbm_prep.transform(validation)
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=350, learning_rate=0.025, num_leaves=15,
            min_child_samples=100, subsample=0.9, colsample_bytree=0.85,
            reg_lambda=5.0, reg_alpha=0.2, random_state=SEED, n_jobs=8, verbose=-1)
        weights = 1 / train["field_size"].to_numpy()
        model.fit(X_train_l, train["won"], sample_weight=weights)
        predictions["validation"][label + "_lgbm"] = normalise_by_race(validation, model.predict_proba(X_val_l)[:, 1])
        predictions["train"][label + "_lgbm"] = normalise_by_race(train, model.predict_proba(X_train_l)[:, 1])
        artifacts["models"][label + "_lgbm"] = (model, lgbm_prep, lgbm_numeric)
        if hasattr(model, "booster_"):
            names = [x.replace("num__", "").replace("cat__", "") for x in lgbm_prep.get_feature_names_out()]
            gains = model.booster_.feature_importance(importance_type="gain")
            artifacts[label + "_lgbm_importance"] = sorted(zip(names, gains.tolist()), key=lambda x: x[1], reverse=True)
        artifacts[label + "_logit_coefficients"] = sorted(zip(logit_features, logit_model.coef_.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return predictions, artifacts


def place_value_bets(frame, probs, edge=0.05, odds_cap=30.0, stake=10.0):
    work = frame[["race_id", "date", "race_no", "horse_no", "win_odds", "won", "win_dividend_hkd10"]].copy()
    work["probability"] = probs
    work["expected_value"] = work["probability"] * work["win_odds"] - 1
    work = work[(work["expected_value"] > edge) & (work["win_odds"] <= odds_cap)]
    if work.empty:
        return work.assign(gross_return=[], net_profit=[])
    work = work.sort_values(["race_id", "expected_value"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
    work = work.sort_values(["date", "race_no"]).reset_index(drop=True)
    work["gross_return"] = np.where(work["won"] == 1, work["win_dividend_hkd10"], 0.0)
    work["net_profit"] = work["gross_return"] - stake
    return work


def favourite_bets(frame, stake=10.0):
    work = frame.sort_values(["race_id", "win_odds", "horse_no"]).groupby("race_id", as_index=False).head(1).copy()
    work = work.sort_values(["date", "race_no"]).reset_index(drop=True)
    work["gross_return"] = np.where(work["won"] == 1, work["win_dividend_hkd10"], 0.0)
    work["net_profit"] = work["gross_return"] - stake
    return work


def top_probability_bets(frame, probs, stake=10.0):
    work = frame[["race_id", "date", "race_no", "horse_no", "win_odds", "won", "win_dividend_hkd10"]].copy()
    work["probability"] = probs
    work = work.sort_values(["race_id", "probability", "horse_no"], ascending=[True, False, True]).groupby("race_id", as_index=False).head(1)
    work = work.sort_values(["date", "race_no"]).reset_index(drop=True)
    work["gross_return"] = np.where(work["won"] == 1, work["win_dividend_hkd10"], 0.0)
    work["net_profit"] = work["gross_return"] - stake
    return work


def betting_summary(bets, frame=None, stake=10.0, bootstrap=3000):
    if bets is None or len(bets) == 0:
        return {"bets": 0}
    profits = bets["net_profit"].to_numpy()
    cum = np.cumsum(profits)
    drawdown = np.maximum.accumulate(np.r_[0, cum])[1:] - cum
    result = {"bets": int(len(bets)), "meetings": int(bets["date"].nunique()),
              "stake_hkd": float(stake * len(bets)), "profit_hkd": float(profits.sum()),
              "roi": float(profits.sum() / (stake * len(bets))),
              "strike_rate": float(bets["won"].mean()), "mean_odds": float(bets["win_odds"].mean()),
              "max_drawdown_hkd": float(max(0, drawdown.max(initial=0)))}
    if bootstrap and len(bets) >= 10:
        grouped = list(bets.groupby("date")["net_profit"].sum().to_numpy())
        values = np.array(grouped)
        samples = np.random.default_rng(SEED).choice(values, size=(bootstrap, len(values)), replace=True).sum(axis=1)
        stakes = bets.groupby("date").size().to_numpy() * stake
        stake_samples = np.random.default_rng(SEED + 1).choice(stakes, size=(bootstrap, len(stakes)), replace=True).sum(axis=1)
        # Resample paired clusters jointly for a conservative meeting-cluster interval.
        idx = np.random.default_rng(SEED + 2).integers(0, len(values), size=(bootstrap, len(values)))
        roi_samples = values[idx].sum(axis=1) / stakes[idx].sum(axis=1)
        result["roi_ci95"] = [float(x) for x in np.quantile(roi_samples, [0.025, 0.975])]
    return result


def paired_metric_intervals(frame, probs_dict, benchmark="Market", bootstrap=5000):
    """Meeting-cluster bootstrap for paired race-metric differences."""
    sizes, starts, winners = race_layout(frame)
    race_dates = frame.groupby("race_id", sort=False)["date"].first().to_numpy()
    unique_dates, date_idx = np.unique(race_dates, return_inverse=True)
    meeting_counts = np.bincount(date_idx, minlength=len(unique_dates)).astype(float)
    idx = np.random.default_rng(SEED + 20).integers(
        0, len(unique_dates), size=(bootstrap, len(unique_dates)))
    denominator = meeting_counts[idx].sum(axis=1)
    reference = np.asarray(probs_dict[benchmark], dtype=float)

    def per_race(probs):
        probs = np.asarray(probs, dtype=float)
        log_loss = -np.log(np.clip(probs[winners], 1e-12, 1))
        brier = np.add.reduceat(probs ** 2, starts) - 2 * probs[winners] + 1
        top1 = np.array([
            float(start + np.argmax(probs[start:start + size]) == winner)
            for start, size, winner in zip(starts, sizes, winners)])
        return {"log_loss": log_loss, "brier": brier, "top1": top1}

    base = per_race(reference)
    rows = []
    for name, probs in probs_dict.items():
        if name in ["Uniform", benchmark]:
            continue
        current = per_race(probs)
        for metric in ["log_loss", "brier", "top1"]:
            delta = current[metric] - base[metric]
            meeting_sums = np.bincount(date_idx, weights=delta, minlength=len(unique_dates))
            samples = meeting_sums[idx].sum(axis=1) / denominator
            rows.append({"model": name, "metric": metric, "difference_vs_market": float(delta.mean()),
                         "ci95_low": float(np.quantile(samples, 0.025)),
                         "ci95_high": float(np.quantile(samples, 0.975)),
                         "better_direction": "negative" if metric in ["log_loss", "brier"] else "positive"})
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df, exclusions = load_and_engineer()
    df.to_csv(DATA / "model_runners.csv", index=False)
    (RESULTS / "sample_exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    print("Clean sample:", df["race_id"].nunique(), "races,", len(df), "runners", flush=True)

    train = df[df["season"].isin(["2021/22", "2022/23", "2023/24"])].copy()
    validation = df[df["season"] == "2024/25"].copy()
    test = df[df["season"] == "2025/26"].copy()
    for frame in [train, validation, test]:
        frame.sort_values(["race_order", "horse_no"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    print("Splits:", [(x["season"].iloc[0], x["race_id"].nunique(), len(x)) for x in [train, validation, test]], flush=True)

    fitted, artifacts = train_models(train, validation)
    validation_probs = {"Uniform": 1.0 / validation["field_size"].to_numpy(dtype=float),
                        "Market": validation["market_prob"].to_numpy(dtype=float)}
    test_probs = {"Uniform": 1.0 / test["field_size"].to_numpy(dtype=float),
                  "Market": test["market_prob"].to_numpy(dtype=float)}
    # Each runner-level baseline vector sums to one within its race.
    temperatures = {}
    for name, (model, prep, features) in artifacts["models"].items():
        if name.endswith("logit"):
            p_test = model.predict_proba(prep.transform(test), test)
        else:
            p_test = normalise_by_race(test, model.predict_proba(prep.transform(test))[:, 1])
        validation_probs[name] = fitted["validation"][name]
        test_probs[name] = p_test
    for name in list(validation_probs):
        if name == "Uniform":
            temperatures[name] = 1.0
        else:
            temperatures[name] = fit_temperature(validation, validation_probs[name])
        validation_probs[name] = temperature_transform(validation, validation_probs[name], temperatures[name])
        test_probs[name] = temperature_transform(test, test_probs[name], temperatures[name])

    max_probability_sum_error = 0.0
    for split, frame, probs_dict in [("validation", validation, validation_probs), ("test", test, test_probs)]:
        for name, probs in probs_dict.items():
            sums = pd.DataFrame({"race_id": frame["race_id"].to_numpy(), "p": probs}).groupby("race_id", sort=False)["p"].sum()
            error = float((sums - 1).abs().max())
            max_probability_sum_error = max(max_probability_sum_error, error)
            if error > 1e-8 or not np.isfinite(probs).all():
                raise ValueError(f"Invalid probabilities for {split}/{name}: sum error {error}")

    metric_rows = []
    for split, frame, probs_dict in [("validation", validation, validation_probs), ("test", test, test_probs)]:
        for name, probs in probs_dict.items():
            metric_rows.append({"split": split, "model": name, "temperature": temperatures[name], **race_metrics(frame, probs)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)

    # Select the primary model using validation log loss only (market and uniform are benchmarks).
    candidate_metrics = metrics[(metrics["split"] == "validation") & (~metrics["model"].isin(["Uniform", "Market"]))]
    primary = candidate_metrics.sort_values("log_loss").iloc[0]["model"]
    (RESULTS / "primary_model.txt").write_text(primary, encoding="utf-8")

    betting_rows = []
    bet_details = {}
    top_bet_details = {}
    for name, probs in test_probs.items():
        if name not in ["Uniform", "Market"]:
            bets = place_value_bets(test, probs)
            bet_details[name] = bets
            betting_rows.append({"strategy": name + " value", **betting_summary(bets)})
            top_bets = top_probability_bets(test, probs)
            top_bet_details[name] = top_bets
            betting_rows.append({"strategy": name + " top pick", **betting_summary(top_bets)})
    favourites = favourite_bets(test)
    betting_rows.append({"strategy": "Favourite every race", **betting_summary(favourites)})
    all_profit = test["win_dividend_hkd10"].where(test["won"] == 1, 0).sum() - 10 * len(test)
    betting_rows.append({"strategy": "Every runner", "bets": int(len(test)), "stake_hkd": float(10 * len(test)),
                         "profit_hkd": float(all_profit), "roi": float(all_profit / (10 * len(test))),
                         "strike_rate": float(test["won"].mean()), "mean_odds": float(test["win_odds"].mean())})
    betting = pd.DataFrame(betting_rows)
    betting.to_csv(RESULTS / "betting_summary.csv", index=False)
    for name, bets in bet_details.items():
        bets.to_csv(RESULTS / f"bets_{name}.csv", index=False)
    for name, bets in top_bet_details.items():
        bets.to_csv(RESULTS / f"top_pick_{name}.csv", index=False)
    favourites.to_csv(RESULTS / "bets_favourite.csv", index=False)

    sensitivity = []
    for edge in [0.0, 0.05, 0.10, 0.20, 0.30]:
        for cap in [10, 20, 30, 1000]:
            bets = place_value_bets(test, test_probs[primary], edge=edge, odds_cap=cap)
            summary = betting_summary(bets, bootstrap=0)
            sensitivity.append({"edge": edge, "odds_cap": cap, **summary})
    pd.DataFrame(sensitivity).to_csv(RESULTS / "betting_sensitivity.csv", index=False)

    # Prediction file and descriptive tables.
    prediction_output = test[["race_id", "date", "season", "race_no", "horse_no", "horse_id", "won", "win_odds", "win_dividend_hkd10"]].copy()
    for name, probs in test_probs.items():
        prediction_output[name] = probs
    prediction_output.to_csv(RESULTS / "test_predictions.csv", index=False)
    paired_metric_intervals(test, test_probs).to_csv(RESULTS / "predictive_comparison_intervals.csv", index=False)
    race_level = df.drop_duplicates("race_id")
    descriptive = race_level.groupby("season").agg(
        races=("race_id", "nunique"), meetings=("date", "nunique"),
        mean_field=("field_size", "mean"), mean_overround=("overround", "mean")).reset_index()
    descriptive["runners"] = descriptive["season"].map(df.groupby("season").size())
    favourites_all = favourite_bets(df)
    descriptive["favourite_win_rate"] = descriptive["season"].map(favourites_all.groupby("season")["won"].mean())
    descriptive = descriptive[["season", "races", "runners", "meetings", "mean_field", "mean_overround", "favourite_win_rate"]]
    descriptive.to_csv(RESULTS / "season_descriptives.csv", index=False)

    test_odds = test.copy()
    test_odds["odds_band"] = pd.cut(test_odds["win_odds"], bins=[1, 2, 4, 7, 12, 25, 50, 100, np.inf],
        labels=["1.0–2.0", "2.1–4.0", "4.1–7.0", "7.1–12.0", "12.1–25.0", "25.1–50.0", "50.1–100", ">100"], include_lowest=True)
    odds_band = test_odds.groupby("odds_band", observed=False).agg(runners=("won", "size"), observed_win_rate=("won", "mean"),
        mean_market_probability=("market_prob", "mean"), mean_odds=("win_odds", "mean")).reset_index()
    returns = test_odds.groupby("odds_band", observed=False).apply(lambda x: (x["win_dividend_hkd10"].where(x["won"] == 1, 0).sum() - 10 * len(x)) / (10 * len(x)), include_groups=False)
    odds_band["roi"] = odds_band["odds_band"].map(returns)
    odds_band.to_csv(RESULTS / "odds_band_analysis.csv", index=False)

    # Reliability curves use common quantile bins within each model.
    reliability_rows = []
    for name, probs in test_probs.items():
        temp = pd.DataFrame({"p": probs, "y": test["won"].to_numpy()})
        temp["bin"] = pd.qcut(temp["p"], 10, duplicates="drop")
        grouped = temp.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), observed=("y", "mean"), count=("y", "size")).reset_index()
        grouped["model"] = name
        reliability_rows.append(grouped)
    pd.concat(reliability_rows, ignore_index=True).to_csv(RESULTS / "reliability.csv", index=False)

    # Cumulative profit is aligned over the complete test-race sequence.
    race_sequence = test[["race_id", "date", "race_no"]].drop_duplicates().sort_values(["date", "race_no"]).reset_index(drop=True)
    for label, bets in [(primary + " value", bet_details[primary]), ("Favourite every race", favourites)]:
        race_sequence[label] = race_sequence["race_id"].map(bets.groupby("race_id")["net_profit"].sum()).fillna(0)
        race_sequence[label + " cumulative"] = race_sequence[label].cumsum()
    race_sequence.to_csv(RESULTS / "cumulative_profit.csv", index=False)

    summary = {"primary_model": primary, "temperatures": temperatures,
               "penalties": artifacts["penalties"],
               "train": {"races": int(train["race_id"].nunique()), "runners": int(len(train))},
               "validation": {"races": int(validation["race_id"].nunique()), "runners": int(len(validation))},
               "test": {"races": int(test["race_id"].nunique()), "runners": int(len(test))},
               "max_probability_sum_error": max_probability_sum_error,
               "sample": exclusions}
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "feature_importance.json").write_text(json.dumps({k: v for k, v in artifacts.items() if k.endswith("importance") or k.endswith("coefficients")}, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print(betting.to_string(index=False), flush=True)
    print("Primary model:", primary, flush=True)


if __name__ == "__main__":
    main()
