#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
供水管网流量数据批处理脚本。

处理逻辑概述：
1) 先做时间对齐和观测层质控；
2) 再估计日基线和漂移；
3) 随后在残差层识别尖峰、跳变和 stuck；
4) 对慢变偏移区分长期缓慢变化（gradual_drift）与平台型偏移（level_shift）；
5) 最后进行保守清洗并输出结果表、故障工作簿、诊断图和汇总。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 可调参数区（默认值 + 说明）
# ==========================
SAMPLING_FREQ = "5min"  # 目标采样间隔：题目要求每5分钟一条
POINTS_PER_DAY = 288  # 24*60/5
STUCK_WINDOW = 8  # 40分钟窗口（8个点）
SHORT_GAP_LIMIT = 3  # <=3点短缺失可插值修复

# spike 检测阈值：残差 MAD 和一阶差分 MAD 的联合判定
SPIKE_MAD_K = 6.0
DIFF_MAD_K = 6.0

# stuck 判定阈值：方差、窗口最大差分、唯一值个数
STUCK_VAR_EPS = 1e-6
STUCK_MAX_DIFF_EPS = 1e-3
STUCK_UNIQUE_MAX = 2

# 跳变替代检测（无ruptures时）前后窗口点数
JUMP_WINDOW = 12  # 1小时窗口
JUMP_DIFF_MAD_K = 5.0

# LOWESS 平滑强度（无statsmodels时退化为滚动中位数）
LOWESS_FRAC = 0.05
DRIFT_ROLLING_WINDOW = 25

# 漂移分类阈值
DRIFT_GRADUAL_SLOPE_K = 2.5  # 斜率MAD倍数阈值，较小斜率视为长期缓慢变化
LEVEL_SHIFT_MIN_POINTS = 12  # 至少1小时
LEVEL_SHIFT_MAG_K = 3.5  # 幅值阈值倍数

# 状态列辅助判定阈值（启发式）
STATUS_ERROR_RATE_THRESHOLD = 0.2
STATUS_QUALITY_BAD_SET = {"bad", "poor", "invalid", "fail", "error", "异常", "错误", "坏"}

# ==========================
# 可选依赖（自动回退）
# ==========================
HAS_STATSMODELS = False
HAS_RUPTURES = False

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    HAS_STATSMODELS = True
except Exception:
    lowess = None

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except Exception:
    rpt = None


# ==========================
# 工具函数
# ==========================
def robust_mad(x: pd.Series) -> float:
    """计算鲁棒 MAD（加上极小值避免除零）。"""
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1e-12
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(mad if mad > 1e-12 else 1e-12)


def keyword_score(name: str, keywords: List[str]) -> int:
    """根据关键词命中次数打分。"""
    n = str(name).strip().lower()
    return sum(1 for k in keywords if k in n)


def contiguous_segments(mask: pd.Series) -> List[Tuple[int, int]]:
    """将布尔序列转为连续段 [start, end]（含端点，按位置索引）。"""
    vals = mask.fillna(False).to_numpy(dtype=bool)
    segs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(vals) - 1))
    return segs


# ==========================
# 主流程函数
# ==========================
def find_input_files(base_dir: Path) -> List[Path]:
    """查找输入文件，优先 `数据/第一题、流量数据/*.xlsx`，否则回退到根目录样本xlsx。"""
    primary = base_dir / "数据" / "第一题、流量数据"
    files: List[Path] = []
    if primary.exists():
        files = sorted([p for p in primary.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        files = sorted([p for p in base_dir.glob("样本*.xlsx") if not p.name.startswith("~$")])
    return files


def detect_columns(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, object]:
    """自动识别时间列、主流量列、状态列。"""
    cols = list(df.columns)

    # 时间列候选
    time_keywords = ["时间", "日期", "datetime", "time", "timestamp"]
    time_candidates = sorted(cols, key=lambda c: keyword_score(str(c), time_keywords), reverse=True)

    best_time_col = None
    best_time_score = -1.0
    for c in time_candidates:
        parsed = pd.to_datetime(df[c], errors="coerce")
        nonnull = parsed.notna().mean()
        keyscore = keyword_score(str(c), time_keywords)
        score = nonnull * 10 + keyscore
        if nonnull > 0.1 and score > best_time_score:
            best_time_score = score
            best_time_col = c

    # 主流量列候选
    flow_keywords = ["流量", "瞬时流量", "流量值", "value", "测量值"]
    status_keywords = ["质量", "错误", "状态", "quality", "error", "status", "io"]

    status_cols = [c for c in cols if keyword_score(str(c), status_keywords) > 0]

    flow_candidates = sorted(cols, key=lambda c: keyword_score(str(c), flow_keywords), reverse=True)
    best_flow_col = None
    best_flow_score = -1.0

    for c in flow_candidates:
        if c == best_time_col:
            continue
        ser = pd.to_numeric(df[c], errors="coerce")
        nonnull = ser.notna().mean()
        var = float(np.nanvar(ser.to_numpy(dtype=float))) if nonnull > 0 else 0.0
        keyscore = keyword_score(str(c), flow_keywords)
        status_penalty = 2.0 if c in status_cols else 0.0
        score = keyscore * 2 + nonnull * 8 + (1.0 if var > 1e-9 else -1.0) - status_penalty
        if score > best_flow_score:
            best_flow_score = score
            best_flow_col = c

    # 若仍不理想，从数值列兜底
    if best_flow_col is None or pd.to_numeric(df[best_flow_col], errors="coerce").notna().mean() < 0.2:
        numeric_cols = []
        for c in cols:
            if c == best_time_col or c in status_cols:
                continue
            ser = pd.to_numeric(df[c], errors="coerce")
            numeric_cols.append((c, ser.notna().mean(), float(np.nanvar(ser.to_numpy(dtype=float)))))
        if numeric_cols:
            numeric_cols.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best_flow_col = numeric_cols[0][0]

    logger.info(
        "列识别结果: time_col=%s, value_col=%s, status_cols=%s",
        best_time_col,
        best_flow_col,
        status_cols,
    )

    return {
        "time_col": best_time_col,
        "value_col": best_flow_col,
        "status_cols": status_cols,
    }


def load_and_prepare_file(file_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, object], str]:
    """读取单个xlsx首个sheet，识别列并构造基础字段。"""
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_name = xls.sheet_names[0]
    df_raw = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
    logger.info("读取文件=%s, sheet=%s, 原始形状=%s", file_path.name, sheet_name, df_raw.shape)

    col_info = detect_columns(df_raw, logger)
    time_col = col_info["time_col"]
    value_col = col_info["value_col"]

    if time_col is None or value_col is None:
        raise ValueError(f"无法识别时间列或主流量列: time_col={time_col}, value_col={value_col}")

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df["raw_value"] = pd.to_numeric(df[value_col], errors="coerce")

    # 时间基础问题标记
    df["is_time_issue"] = df["timestamp"].isna()
    df["is_duplicate_timestamp"] = False
    valid_time = df["timestamp"].notna()
    df.loc[valid_time, "is_duplicate_timestamp"] = df.loc[valid_time, "timestamp"].duplicated(keep=False)

    # 乱序标记
    ts = df["timestamp"]
    df["is_out_of_order"] = ts.notna() & (ts < ts.cummax().shift(1))

    # 状态异常启发式
    status_issue = pd.Series(False, index=df.index)
    for c in col_info["status_cols"]:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            # 错误率类
            name = str(c).lower()
            if ("错误率" in str(c)) or ("error" in name) or ("io" in name):
                status_issue = status_issue | (pd.to_numeric(s, errors="coerce") > STATUS_ERROR_RATE_THRESHOLD)
            else:
                status_issue = status_issue | (pd.to_numeric(s, errors="coerce") > 0)
        else:
            ss = s.astype(str).str.lower()
            status_issue = status_issue | ss.apply(lambda v: any(b in v for b in STATUS_QUALITY_BAD_SET))
    df["is_status_issue"] = status_issue.fillna(False)

    # 保留关键信息
    keep_cols = ["timestamp", "raw_value", "is_time_issue", "is_duplicate_timestamp", "is_out_of_order", "is_status_issue"]
    for c in col_info["status_cols"]:
        if c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols]
    return df, col_info, sheet_name


def align_to_5min_grid(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """对齐到5分钟网格；重复时间按均值聚合并保留标记。"""
    d = df.copy()

    # 只对可解析时间进行网格化
    valid = d["timestamp"].notna()
    d_valid = d.loc[valid].copy()

    if d_valid.empty:
        raise ValueError("没有可解析时间戳，无法对齐。")

    d_valid["timestamp"] = d_valid["timestamp"].dt.floor(SAMPLING_FREQ)

    agg_dict = {
        "raw_value": "mean",
        "is_time_issue": "max",
        "is_duplicate_timestamp": "max",
        "is_out_of_order": "max",
        "is_status_issue": "max",
    }
    status_cols = [c for c in d_valid.columns if c not in agg_dict and c != "timestamp"]
    for c in status_cols:
        # 状态列尽量保留均值
        if pd.api.types.is_numeric_dtype(d_valid[c]):
            agg_dict[c] = "mean"
        else:
            agg_dict[c] = "first"

    grouped = d_valid.groupby("timestamp", as_index=False).agg(agg_dict)

    full_idx = pd.date_range(grouped["timestamp"].min(), grouped["timestamp"].max(), freq=SAMPLING_FREQ)
    aligned = grouped.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()

    # 缺失标记
    aligned["is_missing"] = aligned["raw_value"].isna()
    for c in ["is_time_issue", "is_duplicate_timestamp", "is_out_of_order", "is_status_issue"]:
        aligned[c] = aligned[c].fillna(False).astype(bool)

    logger.info(
        "时间对齐完成: 原有效记录=%d, 对齐后记录=%d, 缺失=%d",
        len(grouped),
        len(aligned),
        int(aligned["is_missing"].sum()),
    )

    return aligned


def build_daily_baseline(df: pd.DataFrame, value_col: str = "raw_value") -> pd.Series:
    """构建288点日基线（中位数模板），并映射到全时序。"""
    d = df.copy()
    d["slot"] = d["timestamp"].dt.hour * 12 + d["timestamp"].dt.minute // 5

    valid_mask = (
        d[value_col].notna()
        & (~d.get("is_status_issue", False))
        & (~d.get("is_spike", False))
        & (~d.get("is_stuck", False))
    )

    template = d.loc[valid_mask].groupby("slot")[value_col].median()

    full_template = pd.Series(index=np.arange(POINTS_PER_DAY), dtype=float)
    full_template.loc[template.index] = template.values
    full_template = full_template.interpolate(limit_direction="both")

    baseline = d["slot"].map(full_template)
    return baseline


def estimate_drift(df: pd.DataFrame, baseline: pd.Series, logger: logging.Logger) -> pd.Series:
    """估计慢变漂移：优先 LOWESS，否则滚动中位数。"""
    raw_minus_base = df["raw_value"] - baseline
    x = np.arange(len(df))

    if HAS_STATSMODELS and lowess is not None:
        valid = raw_minus_base.notna()
        if valid.sum() < 10:
            drift = raw_minus_base.rolling(DRIFT_ROLLING_WINDOW, center=True, min_periods=1).median()
        else:
            y = raw_minus_base[valid].to_numpy(dtype=float)
            xv = x[valid.to_numpy()]
            sm = lowess(y, xv, frac=LOWESS_FRAC, return_sorted=False)
            drift = pd.Series(np.nan, index=df.index, dtype=float)
            drift.loc[valid] = sm
            drift = drift.interpolate(limit_direction="both")
        logger.info("漂移估计使用 LOWESS。")
    else:
        drift = raw_minus_base.rolling(DRIFT_ROLLING_WINDOW, center=True, min_periods=1).median()
        logger.info("漂移估计回退为滚动中位数（未安装statsmodels）。")

    return drift


def compute_residual(df: pd.DataFrame) -> pd.Series:
    """计算残差 = 原始值 - 基线 - 漂移。"""
    return df["raw_value"] - df["baseline"] - df["drift"]


def detect_spikes(df: pd.DataFrame) -> pd.Series:
    """尖峰检测：残差绝对值 + 一阶差分联合 MAD 判定。"""
    r = df["residual"]
    diff = r.diff().abs()

    r_med = r.median(skipna=True)
    r_mad = robust_mad(r)
    d_med = diff.median(skipna=True)
    d_mad = robust_mad(diff)

    cond1 = (r - r_med).abs() > SPIKE_MAD_K * 1.4826 * r_mad
    cond2 = (diff - d_med).abs() > DIFF_MAD_K * 1.4826 * d_mad
    is_spike = (cond1 & cond2).fillna(False)
    return is_spike


def detect_jumps(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.Series, pd.Series]:
    """跳变检测：优先 ruptures PELT；失败时前后窗口中位数差。返回点标记和段ID。"""
    n = len(df)
    is_jump = pd.Series(False, index=df.index)
    seg_id = pd.Series(np.nan, index=df.index)
    sig = df["residual"].interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)

    if HAS_RUPTURES and rpt is not None and n >= 20:
        try:
            model = rpt.Pelt(model="rbf").fit(sig)
            # penalty随样本规模变化，偏保守
            pen = max(5.0, np.log(n) * np.nanstd(sig))
            bkps = model.predict(pen=pen)
            starts = [0] + bkps[:-1]
            sid = 0
            for s, e in zip(starts, bkps):
                if e - s <= 1:
                    continue
                if s > 0:
                    is_jump.iloc[s] = True
                seg_id.iloc[s:e] = sid
                sid += 1
            logger.info("跳变检测使用 ruptures PELT。断点数=%d", len(bkps) - 1)
        except Exception as ex:
            logger.warning("ruptures失败，回退窗口法: %s", ex)
            is_jump, seg_id = _detect_jumps_fallback(df)
    else:
        is_jump, seg_id = _detect_jumps_fallback(df)
        logger.info("跳变检测回退为前后窗口中位数差（未安装ruptures或样本过短）。")

    return is_jump.fillna(False), seg_id


def _detect_jumps_fallback(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    r = df["residual"].copy()
    n = len(r)
    score = pd.Series(np.nan, index=r.index)

    for i in range(JUMP_WINDOW, n - JUMP_WINDOW):
        prev_med = r.iloc[i - JUMP_WINDOW : i].median(skipna=True)
        next_med = r.iloc[i : i + JUMP_WINDOW].median(skipna=True)
        score.iloc[i] = abs(next_med - prev_med)

    mad = robust_mad(score)
    med = score.median(skipna=True)
    is_jump = (score - med) > JUMP_DIFF_MAD_K * 1.4826 * mad
    is_jump = is_jump.fillna(False)

    seg_id = pd.Series(np.nan, index=r.index)
    sid = 0
    for s, e in contiguous_segments(is_jump):
        seg_id.iloc[s : e + 1] = sid
        sid += 1
    return is_jump, seg_id


def detect_stuck(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """stuck检测：窗口方差小 + 窗口最大相邻差小 + 唯一值少。"""
    x = df["raw_value"].copy()
    n = len(x)
    is_stuck = pd.Series(False, index=x.index)

    for i in range(STUCK_WINDOW - 1, n):
        w = x.iloc[i - STUCK_WINDOW + 1 : i + 1]
        if w.isna().any():
            continue
        var_ok = np.var(w.to_numpy(dtype=float)) <= STUCK_VAR_EPS
        max_diff_ok = w.diff().abs().max() <= STUCK_MAX_DIFF_EPS
        uniq_ok = w.nunique(dropna=True) <= STUCK_UNIQUE_MAX
        # 至少满足两个条件 + var条件必须成立，避免误伤真实平稳段
        if var_ok and ((max_diff_ok and uniq_ok) or (max_diff_ok and var_ok) or (uniq_ok and var_ok)):
            is_stuck.iloc[i - STUCK_WINDOW + 1 : i + 1] = True

    seg_id = pd.Series(np.nan, index=x.index)
    sid = 0
    for s, e in contiguous_segments(is_stuck):
        seg_id.iloc[s : e + 1] = sid
        sid += 1

    return is_stuck, seg_id


def classify_drift_segments(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """对漂移项分类：gradual_drift 与 level_shift。"""
    drift = df["drift"].copy()
    drift_diff = drift.diff()

    slope_mad = robust_mad(drift_diff)
    slope_thr = DRIFT_GRADUAL_SLOPE_K * 1.4826 * slope_mad

    # gradual: 小斜率且连续
    gradual_mask = drift_diff.abs() <= slope_thr
    gradual_mask = gradual_mask.fillna(False)

    # level_shift: 通过跳变后平台抬升/降低特征识别
    d_mad = robust_mad(drift)
    level_thr = LEVEL_SHIFT_MAG_K * 1.4826 * d_mad
    level_mask = pd.Series(False, index=df.index)

    for i in range(JUMP_WINDOW, len(df) - JUMP_WINDOW):
        left = drift.iloc[i - JUMP_WINDOW : i].median(skipna=True)
        right = drift.iloc[i : i + JUMP_WINDOW].median(skipna=True)
        if abs(right - left) > level_thr:
            level_mask.iloc[i : i + LEVEL_SHIFT_MIN_POINTS] = True

    # 仅保留足够长的平台段
    for s, e in contiguous_segments(level_mask):
        if (e - s + 1) < LEVEL_SHIFT_MIN_POINTS:
            level_mask.iloc[s : e + 1] = False

    # gradual 排除 level_shift 段
    gradual_mask = gradual_mask & (~level_mask)

    seg_id = pd.Series(np.nan, index=df.index)
    sid = 0
    for s, e in contiguous_segments(gradual_mask | level_mask):
        seg_id.iloc[s : e + 1] = sid
        sid += 1

    return gradual_mask.fillna(False), level_mask.fillna(False), seg_id


def conservative_clean(df: pd.DataFrame) -> pd.DataFrame:
    """保守清洗：观测故障修复 + 分析层去 gradual_drift 保留 level_shift。"""
    out = df.copy()

    out["is_observation_anomaly"] = (
        out["is_missing"]
        | out["is_time_issue"]
        | out["is_status_issue"]
        | out["is_spike"]
        | out["is_stuck"]
    )

    out["is_candidate_physical_event"] = out["is_jump"] | out["level_shift"]

    # 第一层：只处理明确观测故障，不强修长段
    c1 = out["raw_value"].copy()
    c1[out["is_observation_anomaly"]] = np.nan

    # 仅短缺失插值
    miss = c1.isna()
    for s, e in contiguous_segments(miss):
        seg_len = e - s + 1
        if seg_len <= SHORT_GAP_LIMIT:
            c1.iloc[s : e + 1] = np.nan
        else:
            # 长段保留NaN
            continue
    c1 = c1.interpolate(limit=SHORT_GAP_LIMIT, limit_direction="both")

    out["cleaned_observation_value"] = c1

    # 第二层：分析值去 gradual_drift，但保留 level_shift
    c2 = c1.copy()
    c2 = c2 - out["gradual_drift"].fillna(0.0)
    out["cleaned_analysis_value"] = c2

    return out


def make_plots(df: pd.DataFrame, out_png: Path, file_label: str) -> None:
    """生成4子图诊断图。"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    t = df["timestamp"]
    raw = df["raw_value"]

    # 1) 原始 vs 清洗
    axes[0].plot(t, raw, label="raw", color="steelblue", linewidth=1)
    axes[0].plot(t, df["cleaned_observation_value"], label="cleaned_obs", color="orange", linewidth=1)
    axes[0].set_title(f"{file_label} - 原始与清洗")
    axes[0].legend(loc="upper right")

    # 2) 基线 + 核心漂移（避免过载）
    axes[1].plot(t, df["baseline"], label="baseline", color="green", linewidth=1)
    axes[1].plot(t, df["drift"], label="drift", color="purple", linewidth=1)
    ls_mask = df["level_shift"].fillna(False)
    if ls_mask.any():
        axes[1].scatter(t[ls_mask], df.loc[ls_mask, "drift"], s=8, color="red", label="level_shift")
    axes[1].set_title("日基线与核心漂移")
    axes[1].legend(loc="upper right")

    # 3) 残差
    axes[2].plot(t, df["residual"], color="gray", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("残差序列")

    # 4) 故障分布（直接用raw_value作纵轴，不铺残差背景）
    axes[3].plot(t, raw, color="lightgray", linewidth=0.8, label="raw")
    fault_types = [
        ("is_spike", "spike", "red"),
        ("is_jump", "jump", "blue"),
        ("is_stuck", "stuck", "black"),
        ("gradual_drift", "gradual_drift", "orange"),
        ("level_shift", "level_shift", "purple"),
    ]
    for col, name, color in fault_types:
        m = df[col].fillna(False)
        if m.any():
            axes[3].scatter(t[m], raw[m], s=10, color=color, label=name)
    axes[3].set_title("故障点分布（基于原始值）")
    axes[3].legend(loc="upper right", ncol=3, fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    file_path: Path,
    out_dir: Path,
    col_info: Dict[str, object],
) -> Dict[str, object]:
    """保存 per-file CSV、fault workbook、图；返回汇总信息。"""
    sample_name = file_path.stem

    per_file_csv = out_dir / "per_file_csv" / f"{sample_name}_result.csv"
    fault_xlsx = out_dir / "fault_workbooks" / f"{sample_name}_faults.xlsx"
    fig_png = out_dir / "figures" / f"{sample_name}_overview.png"

    for d in [per_file_csv.parent, fault_xlsx.parent, fig_png.parent]:
        d.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["source_file"] = file_path.name

    required_cols = [
        "source_file",
        "timestamp",
        "raw_value",
        "baseline",
        "drift",
        "gradual_drift",
        "level_shift",
        "residual",
        "is_missing",
        "is_duplicate_timestamp",
        "is_time_issue",
        "is_status_issue",
        "is_spike",
        "is_jump",
        "is_stuck",
        "is_observation_anomaly",
        "is_candidate_physical_event",
        "cleaned_observation_value",
        "cleaned_analysis_value",
    ]

    # 确保列存在
    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan

    out[required_cols].to_csv(per_file_csv, index=False, encoding="utf-8-sig")

    # 组织故障sheet
    def _fault_df(mask_col: str, fault_type: str) -> pd.DataFrame:
        tmp = out.loc[out[mask_col].fillna(False), ["timestamp", "raw_value", "baseline", "drift", "residual"]].copy()
        tmp["fault_type"] = fault_type
        tmp["segment_id"] = np.nan
        return tmp

    spike_points = _fault_df("is_spike", "spike")
    jump_points = _fault_df("is_jump", "jump")
    stuck_points = _fault_df("is_stuck", "stuck")
    gradual_points = _fault_df("gradual_drift", "gradual_drift")
    level_points = _fault_df("level_shift", "level_shift")
    time_issue_points = out.loc[
        out["is_missing"].fillna(False) | out["is_time_issue"].fillna(False),
        ["timestamp", "raw_value", "baseline", "drift", "residual", "is_missing", "is_time_issue", "is_duplicate_timestamp"],
    ].copy()

    # 附加状态列
    for s_col in col_info.get("status_cols", []):
        if s_col in out.columns:
            for tdf in [
                spike_points,
                jump_points,
                stuck_points,
                gradual_points,
                level_points,
                time_issue_points,
            ]:
                tdf[s_col] = out.loc[tdf.index, s_col]

    seg_summary = pd.DataFrame(
        {
            "metric": [
                "spike_points",
                "jump_segments",
                "stuck_segments",
                "gradual_drift_segments",
                "level_shift_segments",
            ],
            "value": [
                int(out["is_spike"].sum()),
                len(contiguous_segments(out["is_jump"])),
                len(contiguous_segments(out["is_stuck"])),
                len(contiguous_segments(out["gradual_drift"])),
                len(contiguous_segments(out["level_shift"])),
            ],
        }
    )

    with pd.ExcelWriter(fault_xlsx, engine="openpyxl") as writer:
        spike_points.to_excel(writer, sheet_name="spike_points", index=False)
        jump_points.to_excel(writer, sheet_name="jump_points", index=False)
        stuck_points.to_excel(writer, sheet_name="stuck_points", index=False)
        gradual_points.to_excel(writer, sheet_name="gradual_drift_points", index=False)
        level_points.to_excel(writer, sheet_name="level_shift_points", index=False)
        time_issue_points.to_excel(writer, sheet_name="time_or_missing_issues", index=False)
        seg_summary.to_excel(writer, sheet_name="segment_summary", index=False)

    make_plots(out, fig_png, sample_name)

    summary = {
        "文件名": file_path.name,
        "原始记录数": int(len(df)),
        "时间对齐后记录数": int(len(df)),
        "缺失点数": int(df["is_missing"].sum()),
        "重复点数": int(df["is_duplicate_timestamp"].sum()),
        "状态异常点数": int(df["is_status_issue"].sum()),
        "尖峰点数": int(df["is_spike"].sum()),
        "跳变段数": len(contiguous_segments(df["is_jump"])),
        "stuck段数": len(contiguous_segments(df["is_stuck"])),
        "gradual_drift段数": len(contiguous_segments(df["gradual_drift"])),
        "level_shift段数": len(contiguous_segments(df["level_shift"])),
        "最终可用点数": int(df["cleaned_analysis_value"].notna().sum()),
        "自动识别到的时间列名": str(col_info.get("time_col")),
        "自动识别到的主流量列名": str(col_info.get("value_col")),
    }
    return summary


def configure_logger(log_file: Path) -> logging.Logger:
    """配置日志到文件+控制台。"""
    logger = logging.getLogger("flow_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def main() -> None:
    """程序入口。"""
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "输出" / "第一题流量处理结果"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(out_dir / "run.log")
    logger.info("脚本启动。statsmodels=%s, ruptures=%s", HAS_STATSMODELS, HAS_RUPTURES)

    files = find_input_files(base_dir)
    if not files:
        logger.error("未找到输入xlsx文件，请检查目录结构。")
        print("共处理0个文件，成功0个，失败0个")
        return

    summaries: List[Dict[str, object]] = []
    success = 0
    failed = 0

    for file_path in files:
        try:
            logger.info("开始处理: %s", file_path)
            df0, col_info, sheet = load_and_prepare_file(file_path, logger)
            logger.info("文件=%s 实际读取sheet=%s", file_path.name, sheet)

            df = align_to_5min_grid(df0, logger)

            # 第一轮基线与漂移
            df["baseline"] = build_daily_baseline(df, "raw_value")
            df["drift"] = estimate_drift(df, df["baseline"], logger)

            # 重估更干净基线
            df["residual"] = compute_residual(df)
            df["is_spike"] = detect_spikes(df)
            df["is_stuck"], df["stuck_segment_id"] = detect_stuck(df)
            df["baseline"] = build_daily_baseline(df, "raw_value")
            df["drift"] = estimate_drift(df, df["baseline"], logger)
            df["residual"] = compute_residual(df)

            # 异常检测
            df["is_spike"] = detect_spikes(df)
            df["is_jump"], df["jump_segment_id"] = detect_jumps(df, logger)
            df["is_stuck"], df["stuck_segment_id"] = detect_stuck(df)

            # 漂移分类
            df["gradual_drift"], df["level_shift"], df["drift_segment_id"] = classify_drift_segments(df)

            # 保守清洗
            df = conservative_clean(df)

            summary = save_outputs(df, file_path, out_dir, col_info)
            summaries.append(summary)

            success += 1
            logger.info("处理完成: %s", file_path.name)
        except Exception as ex:
            failed += 1
            logger.exception("处理失败: %s, error=%s", file_path.name, ex)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    logger.info("全部处理结束: total=%d, success=%d, failed=%d", len(files), success, failed)
    print(f"共处理{len(files)}个文件，成功{success}个，失败{failed}个")


if __name__ == "__main__":
    main()
