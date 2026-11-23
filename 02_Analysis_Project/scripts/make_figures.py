#!/usr/bin/env python3
"""Render consistent static ECharts figures from verified analysis outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ECHARTS = ROOT / "scripts" / "vendor" / "echarts.min.js"
FIGURES.mkdir(parents=True, exist_ok=True)

NAVY = "#173F5F"
TEAL = "#1B998B"
GOLD = "#E9C46A"
ORANGE = "#E76F51"
RED = "#C94452"
PURPLE = "#7561A8"
GREY = "#6B7280"
LIGHT = "#D8E0E7"
TEXT = "#24313D"
MODEL_NAMES = {
    "Uniform": "Uniform", "Market": "Market",
    "fundamental_logit": "Fundamental logit", "fundamental_lgbm": "Fundamental LightGBM",
    "combined_logit": "Combined logit", "combined_lgbm": "Combined LightGBM",
}
FEATURE_NAMES = {
    "horse_last3_finish": "Recent form: last three",
    "horse_last_finish": "Most recent finish",
    "jockey_mean_finish": "Jockey mean finish",
    "jockey_place_rate": "Jockey place rate",
    "draw_ratio": "Normalized draw",
    "jockey_win_rate": "Jockey win rate",
    "horse_dist_mean_finish": "Horse distance record",
    "horse_weight_rel": "Relative body weight",
    "field_size": "Field size",
    "horse_course_mean_finish": "Horse course record",
    "horse_starts_log": "Horse experience",
    "actual_weight_rel": "Relative carried weight",
    "trainer_win_rate": "Trainer win rate",
    "draw_x_hv": "Draw × Happy Valley",
    "trainer_place_rate": "Trainer place rate",
    "log_market_prob": "Log market probability",
    "log_win_odds": "Log WIN odds",
    "market_rank": "Market rank",
    "overround": "Market overround",
    "horse_mean_finish": "Horse career finish",
    "horse_place_rate": "Horse place rate",
    "horse_win_rate": "Horse win rate",
    "horse_days_log": "Days since previous start",
    "horse_dist_place_rate": "Horse distance place rate",
    "trainer_mean_finish": "Trainer mean finish",
}


def base(title: str, subtitle: str = "") -> dict:
    return {
        "backgroundColor": "#FFFFFF", "animation": False,
        "title": {"text": title, "subtext": subtitle, "left": 24, "top": 16,
                  "textStyle": {"fontSize": 23, "fontWeight": 700, "color": TEXT},
                  "subtextStyle": {"fontSize": 13, "color": GREY}},
        "textStyle": {"fontFamily": "DejaVu Sans, Arial, sans-serif", "color": TEXT},
        "tooltip": {"trigger": "item"},
    }


def render(page, name: str, option: dict, width: int = 1200, height: int = 700) -> None:
    option = base(**option.pop("_title")) | option
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<style>html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden;background:white}}
#chart{{width:{width}px;height:{height}px}}</style></head><body>
<div id='chart'></div><script src='{ECHARTS.as_uri()}'></script>
<script>const option={json.dumps(option, ensure_ascii=False)};
const chart=echarts.init(document.getElementById('chart'),null,{{renderer:'canvas'}});
chart.setOption(option);window.chartReady=true;</script></body></html>"""
    target = FIGURES / f"_{name}.html"
    target.write_text(html, encoding="utf-8")
    page.set_viewport_size({"width": width, "height": height})
    page.goto(target.as_uri(), wait_until="load")
    page.wait_for_function("window.chartReady === true")
    page.locator("#chart").screenshot(path=str(FIGURES / f"{name}.png"))
    target.unlink()


def season_chart(page) -> None:
    d = pd.read_csv(RESULTS / "season_descriptives.csv")
    colors = [NAVY, NAVY, NAVY, GOLD, TEAL]
    bars = [{"value": round(x), "itemStyle": {"color": c}} for x, c in zip(d.races, colors)]
    option = {"_title": {"title": "Modeling sample by season", "subtitle": "Training (navy), validation (gold), and untouched test (teal); 88 result meetings per season"},
        "legend": {"top": 82, "right": 30, "data": ["Races", "Starters"]},
        "grid": {"left": 80, "right": 86, "top": 132, "bottom": 72},
        "xAxis": {"type": "category", "data": d.season.tolist(), "axisTick": {"show": False}},
        "yAxis": [{"type": "value", "name": "Races", "min": 0, "max": 1000, "splitLine": {"lineStyle": {"color": "#EDF1F4"}}},
                  {"type": "value", "name": "Starters", "min": 0, "max": 14000, "splitLine": {"show": False}}],
        "series": [{"name": "Races", "type": "bar", "data": bars, "barWidth": "46%",
                    "label": {"show": True, "position": "top", "fontWeight": 600}},
                   {"name": "Starters", "type": "line", "yAxisIndex": 1, "data": d.runners.astype(int).tolist(),
                    "symbolSize": 10, "lineStyle": {"width": 3, "color": ORANGE}, "itemStyle": {"color": ORANGE},
                    "label": {"show": True, "position": "bottom", "distance": 12, "color": TEXT,
                              "backgroundColor": "rgba(255,255,255,0.88)", "padding": [2, 4], "borderRadius": 3}}]}
    render(page, "season_sample", option)


def model_chart(page) -> None:
    d = pd.read_csv(RESULTS / "model_metrics.csv")
    order = ["Uniform", "fundamental_logit", "fundamental_lgbm", "combined_logit", "combined_lgbm", "Market"]
    val = d[d.split == "validation"].set_index("model")
    test = d[d.split == "test"].set_index("model")
    option = {"_title": {"title": "Race log loss: validation and held-out test", "subtitle": "Lower is better; axis starts at 1.75 to reveal small differences; tuning uses validation only"},
        "legend": {"top": 82, "right": 30, "data": ["Validation", "Test"]},
        "grid": {"left": 205, "right": 78, "top": 132, "bottom": 66},
        "xAxis": {"type": "value", "name": "Race log loss", "nameLocation": "middle", "nameGap": 40, "min": 1.75, "max": 2.65,
                  "splitLine": {"lineStyle": {"color": "#EDF1F4"}}},
        "yAxis": {"type": "category", "inverse": True, "data": [MODEL_NAMES[x] for x in order],
                  "axisTick": {"show": False}, "axisLabel": {"fontSize": 14}},
        "series": [{"name": "Validation", "type": "bar", "barWidth": 15,
                    "data": [round(val.loc[x, "log_loss"], 4) for x in order],
                    "itemStyle": {"color": GOLD}, "label": {"show": True, "position": "right", "fontSize": 12}},
                   {"name": "Test", "type": "bar", "barWidth": 15,
                    "data": [round(test.loc[x, "log_loss"], 4) for x in order],
                    "itemStyle": {"color": TEAL}, "label": {"show": True, "position": "right", "fontSize": 12}}]}
    render(page, "model_comparison", option)


def reliability_chart(page) -> None:
    d = pd.read_csv(RESULTS / "reliability.csv")
    selected = ["Market", "fundamental_lgbm", "combined_logit"]
    colors = [NAVY, ORANGE, TEAL]
    series = [{"name": "Perfect calibration", "type": "line", "data": [[0, 0], [35, 35]],
               "lineStyle": {"type": "dashed", "width": 2, "color": GREY}, "itemStyle": {"color": GREY},
               "symbol": "none", "silent": True}]
    for name, color in zip(selected, colors):
        x = d[d.model == name]
        series.append({"name": MODEL_NAMES[name], "type": "line", "smooth": False,
                       "data": [[round(100 * a, 3), round(100 * b, 3)] for a, b in zip(x.mean_predicted, x.observed)],
                       "lineStyle": {"width": 3, "color": color}, "itemStyle": {"color": color}, "symbolSize": 9})
    option = {"_title": {"title": "Held-out reliability in probability deciles", "subtitle": "Runner-level quantile bins; axes show winning probability (%)"},
        "legend": {"top": 82, "right": 26},
        "grid": {"left": 84, "right": 50, "top": 142, "bottom": 76},
        "xAxis": {"type": "value", "name": "Mean predicted probability (%)", "nameLocation": "middle", "nameGap": 42, "min": 0, "max": 35,
                  "splitLine": {"lineStyle": {"color": "#EDF1F4"}}},
        "yAxis": {"type": "value", "name": "Observed win rate (%)", "min": 0, "max": 35,
                  "splitLine": {"lineStyle": {"color": "#EDF1F4"}}}, "series": series}
    render(page, "reliability", option)


def odds_chart(page) -> None:
    d = pd.read_csv(RESULTS / "odds_band_analysis.csv")
    labels = d.odds_band.tolist()
    option = {"_title": {"title": "Market calibration and flat-stake returns by WIN-odds band", "subtitle": "Untouched 2025/26 test season; returns settle winning bets using official HK$10 dividends"},
        "legend": {"top": 82, "right": 24, "data": ["Observed win rate", "Mean normalized market probability", "Bet every runner ROI"]},
        "grid": [{"left": 82, "right": 42, "top": 145, "height": 205},
                 {"left": 82, "right": 42, "top": 450, "height": 165}],
        "xAxis": [{"type": "category", "gridIndex": 0, "data": labels, "axisLabel": {"rotate": 25}, "axisTick": {"show": False}},
                  {"type": "category", "gridIndex": 1, "data": labels, "axisLabel": {"rotate": 25}, "axisTick": {"show": False}}],
        "yAxis": [{"type": "value", "gridIndex": 0, "name": "Probability (%)", "min": 0, "max": 60,
                   "splitLine": {"lineStyle": {"color": "#EDF1F4"}}},
                  {"type": "value", "gridIndex": 1, "name": "ROI (%)", "min": -100, "max": 20,
                   "splitLine": {"lineStyle": {"color": "#EDF1F4"}}}],
        "series": [{"name": "Observed win rate", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
                    "data": (100 * d.observed_win_rate).round(2).tolist(), "lineStyle": {"width": 3, "color": TEAL},
                    "itemStyle": {"color": TEAL}, "symbolSize": 8},
                   {"name": "Mean normalized market probability", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
                    "data": (100 * d.mean_market_probability).round(2).tolist(), "lineStyle": {"width": 3, "color": NAVY},
                    "itemStyle": {"color": NAVY}, "symbolSize": 8},
                   {"name": "Bet every runner ROI", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1,
                    "data": [{"value": round(100 * x, 1), "itemStyle": {"color": RED if x < 0 else TEAL}} for x in d.roi],
                    "barWidth": "52%", "label": {"show": True, "position": "bottom", "fontSize": 11}}]}
    render(page, "odds_bands", option, height=720)


def importance_chart(page) -> None:
    d = json.loads((RESULTS / "feature_importance.json").read_text())
    grids, xaxes, yaxes, series = [], [], [], []
    for idx, (key, title, color) in enumerate([
            ("fundamental_lgbm_importance", "Fundamental LightGBM", ORANGE),
            ("combined_lgbm_importance", "Combined LightGBM", NAVY)]):
        rows = d[key]
        total = sum(x[1] for x in rows)
        top = rows[:12][::-1]
        left = 250 if idx == 0 else 1050
        grids.append({"left": left, "right": 890 if idx == 0 else 65, "top": 150, "bottom": 82})
        xaxes.append({"type": "value", "gridIndex": idx, "name": "Share of total gain (%)",
                      "nameLocation": "middle", "nameGap": 42, "min": 0,
                      "splitLine": {"lineStyle": {"color": "#EDF1F4"}}})
        yaxes.append({"type": "category", "gridIndex": idx,
                      "data": [FEATURE_NAMES.get(x[0], x[0].replace("_", " ")) for x in top],
                      "axisTick": {"show": False}, "axisLabel": {"fontSize": 12}})
        series.append({"name": title, "type": "bar", "xAxisIndex": idx, "yAxisIndex": idx,
                       "data": [round(100 * x[1] / total, 2) for x in top], "barWidth": 15,
                       "itemStyle": {"color": color}, "label": {"show": True, "position": "right", "fontSize": 11}})
    option = {"_title": {"title": "LightGBM feature importance", "subtitle": "Gain contribution; market variables enter only the combined specification"},
        "graphic": [{"type": "text", "left": 400, "top": 105, "style": {"text": "Fundamental LightGBM", "fontSize": 17, "fontWeight": 700, "fill": ORANGE}},
                    {"type": "text", "left": 1190, "top": 105, "style": {"text": "Combined LightGBM", "fontSize": 17, "fontWeight": 700, "fill": NAVY}}],
        "grid": grids, "xAxis": xaxes, "yAxis": yaxes, "series": series}
    render(page, "feature_importance", option, width=1600, height=720)


def profit_chart(page) -> None:
    d = pd.read_csv(RESULTS / "cumulative_profit.csv")
    option = {"_title": {"title": "Cumulative flat-stake profit over the test season", "subtitle": "HK$10 per selected bet; no bet is placed in races without a qualifying value selection"},
        "legend": {"top": 82, "right": 24},
        "grid": {"left": 88, "right": 42, "top": 140, "bottom": 76},
        "xAxis": {"type": "category", "data": list(range(len(d))), "name": "Chronological test-race sequence",
                  "nameLocation": "middle", "nameGap": 38,
                  "axisLabel": {"show": False}, "axisTick": {"show": False}, "boundaryGap": False},
        "yAxis": {"type": "value", "name": "Cumulative profit (HK$)", "splitLine": {"lineStyle": {"color": "#EDF1F4"}}},
        "series": [{"name": "Combined logit value", "type": "line", "data": d["combined_logit value cumulative"].round(2).tolist(),
                    "showSymbol": False, "lineStyle": {"width": 3, "color": TEAL}, "itemStyle": {"color": TEAL},
                    "markLine": {"silent": True, "symbol": "none", "data": [{"yAxis": 0}], "lineStyle": {"color": GREY, "type": "dashed"}, "label": {"show": False}}},
                   {"name": "Favourite every race", "type": "line", "data": d["Favourite every race cumulative"].round(2).tolist(),
                    "showSymbol": False, "lineStyle": {"width": 2.5, "color": RED}, "itemStyle": {"color": RED}}]}
    render(page, "cumulative_profit", option)


def sensitivity_chart(page) -> None:
    d = pd.read_csv(RESULTS / "betting_sensitivity.csv")
    edges = [0.0, 0.05, 0.10, 0.20, 0.30]
    caps = [10, 20, 30, 1000]
    data = []
    for yi, edge in enumerate(edges):
        for xi, cap in enumerate(caps):
            row = d[(np.isclose(d.edge, edge)) & (d.odds_cap == cap)].iloc[0]
            if row.bets == 0:
                data.append({"value": [xi, yi, 0], "label": {"formatter": "No bets", "color": GREY}, "itemStyle": {"color": "#F3F4F6"}})
            else:
                roi = 100 * row.roi
                data.append({"value": [xi, yi, float(np.clip(roi, -30, 30))],
                             "label": {"formatter": f"{roi:+.1f}%\nn={int(row.bets)}"}})
    option = {"_title": {"title": "Value-strategy sensitivity to edge and odds cap", "subtitle": "Labels show test ROI and number of bets; colour scale is truncated at ±30%"},
        "grid": {"left": 110, "right": 130, "top": 145, "bottom": 80},
        "xAxis": {"type": "category", "data": ["10", "20", "30", "No cap"], "name": "Maximum decimal WIN odds",
                  "nameLocation": "middle", "nameGap": 42, "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": ["0%", "5%", "10%", "20%", "30%"], "name": "Minimum expected-value edge",
                  "nameLocation": "middle", "nameGap": 72, "nameRotate": 90, "splitArea": {"show": True}},
        "visualMap": {"min": -30, "max": 30, "calculable": False, "orient": "vertical", "right": 25, "top": "middle",
                      "text": ["Gain", "Loss"], "inRange": {"color": [RED, "#F7E5D8", "#FFFFFF", "#D9EFE8", TEAL]}},
        "series": [{"type": "heatmap", "data": data, "label": {"show": True, "fontSize": 14, "fontWeight": 600},
                    "itemStyle": {"borderColor": "#FFFFFF", "borderWidth": 4}, "emphasis": {"disabled": True}}]}
    render(page, "betting_sensitivity", option)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(device_scale_factor=2)
        for function in [season_chart, model_chart, reliability_chart, odds_chart,
                         importance_chart, profit_chart, sensitivity_chart]:
            function(page)
            print(f"Rendered {function.__name__}", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
