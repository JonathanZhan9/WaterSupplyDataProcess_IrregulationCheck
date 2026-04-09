#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
供水管网流量数据批处理脚本
处理逻辑：先做时间对齐和观测层质控，再估计日基线和漂移，随后在残差层识别尖峰、跳变和stuck，
进一步把慢变偏移拆分为 gradual_drift 和 level_shift，最后进行多策略清洗并输出结果。
"""

from __future__ import annotations

import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 可调参数区（统一管理）
# =========================
SAMPLE_INTERVAL = "5min"
STUCK_WINDOW = 8
SHORT_GAP_MAX = 3
SPIKE_MAD_K = 6.0
DIFF_MAD_K = 6.0
STUCK_VAR_THR = 1e-3
STUCK_MAX_DIFF_THR = 1e-3
STUCK_UNIQUE_MAX = 2
JUMP_WINDOW = 6
JUMP_K = 8.0
JUMP_MIN_ABS = 25.0  # 跳变最小绝对幅值，避免把日内周期波动当作跳变
JUMP_MIN_DISTANCE = 24  # 相邻跳变最小间隔（点），防止全段密集误报
JUMP_MARK_HALF_WIDTH = 2  # 跳变点两侧标记宽度
LOWESS_FRAC = 0.04
DRIFT_ROLLING = 288
TIME_PARSE_CANDIDATE_ROWS = 20
CLEAR_PREVIOUS_OUTPUT = True

# drift 分段分类参数
DRIFT_SEGMENT_MAD_K = 3.0  # 漂移候选阈值倍数（基于drift_raw MAD）
DRIFT_MERGE_GAP = 6  # 漂移候选点合并间隙（点）
PLATEAU_MIN_LEN = 24  # 判定平台段最短长度（点）
GRADUAL_MIN_LEN = 24  # 判定渐进漂移最短长度（点）
EDGE_WINDOW = 6  # 计算段边界跳变量时使用的窗口长度
PLATEAU_SLOPE_THR = 0.12  # 平台内部斜率阈值（单位/点）
PLATEAU_EDGE_JUMP_K = 1.2  # 边界跳变需超过幅值比例
DRIFT_ABS_MIN = 5.0  # 漂移绝对幅值最小门槛
PLATEAU_MIN_AMPLITUDE = 8.0  # 平台型偏移最小幅值
PLATEAU_FLATNESS_THR = 0.45  # 平顶比例阈值（越高越平）
PLATEAU_EDGE_RATIO_THR = 1.2  # 边缘突变强度/中部变化强度阈值
GRADUAL_SLOPE_MIN = 0.03  # 判为gradual时的最小斜率阈值（单位/点）
DRIFT_SEGMENT_MERGE_GAP = 6  # 与 DRIFT_MERGE_GAP 同义保留，便于调参
STUCK_SUPPRESS_DRIFT = True  # stuck主导区段抑制漂移分型
ANCHOR_REF_DAYS = 3  # anchored_drift参考期天数
LEVEL_RETURN_WIN = 24  # level_shift下降沿判定窗口
LEVEL_RETURN_HOLD = 6  # 连续满足回落条件的窗口数
LEVEL_RETURN_DROP_RATIO = 0.55  # 相对平台幅值的回落比例阈值
STUCK_GUARD_POINTS = 6  # stuck周围保护区，保护区内不判level_shift
LEVEL_EDGE_WIN = 12  # level边沿检测窗口（点）
LEVEL_RETURN_TOL_RATIO = 0.45  # level回落后与基线一致性的容差比例
GRADUAL_ONSET_BACKTRACK = 5  # gradual起点向前回溯，减少起点突变
GRADUAL_LONG_WIN = 288  # gradual长期趋势观察窗口（点）
GRADUAL_TOTAL_CHANGE_MIN = 30.0  # gradual长期净变化最小幅值
GRADUAL_MONO_RATIO_MIN = 0.72  # gradual整体单调趋势占比下限
GRADUAL_MONO_TOL_K = 0.25  # gradual单调性容差（相对局部噪声）
GRADUAL_ONSET_FRAC = 0.3  # gradual起点阈值占总变化比例
GRADUAL_ONSET_ABS_MIN = 30.0  # gradual起点最小绝对变化阈值

TIME_KEYWORDS = ["时间", "日期", "datetime", "time", "timestamp"]
FLOW_KEYWORDS = ["流量", "瞬时流量", "流量值", "value", "测量值"]
STATUS_KEYWORDS = ["数据质量", "io错误率", "质量", "错误", "状态", "quality", "error", "status"]

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("flow_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def robust_mad(x: pd.Series | np.ndarray) -> float:
    arr = pd.Series(x).dropna().to_numpy()
    if len(arr) == 0:
        return 1e-12
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(float(mad), 1e-12)


def contiguous_segments(mask: pd.Series) -> List[Tuple[int, int]]:
    arr = mask.fillna(False).to_numpy(dtype=bool)
    segs, start = [], None
    for i, flag in enumerate(arr):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(arr) - 1))
    return segs


def merge_segments(segs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not segs:
        return []
    segs = sorted(segs)
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= max_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def find_input_files(base_dir: Path) -> List[Path]:
    primary = base_dir / "数据" / "第一题、流量数据"
    if primary.exists():
        files = sorted(primary.glob("*.xlsx"))
        if files:
            return files
    return sorted(base_dir.glob("样本*.xlsx"))


def detect_columns(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, object]:
    cols = [str(c).strip() for c in df.columns]

    time_candidates = [c for c in cols if any(k in c.lower() for k in TIME_KEYWORDS)]
    if not time_candidates:
        time_candidates = cols[:]

    best_time, best_score = None, -1.0
    for c in time_candidates:
        s = df[c]
        parsed = pd.to_datetime(s, errors="coerce")
        score = 0.7 * parsed.notna().mean() + 0.3 * s.notna().mean()
        if score > best_score:
            best_score = score
            best_time = c

    status_cols = [c for c in cols if any(k in c.lower() for k in STATUS_KEYWORDS)]
    flow_candidates = [c for c in cols if any(k in c.lower() for k in FLOW_KEYWORDS)]

    if flow_candidates:
        flow_col = max(flow_candidates, key=lambda c: pd.to_numeric(df[c], errors="coerce").notna().mean())
    else:
        best_flow, best_flow_score = None, -1.0
        for c in cols:
            if c == best_time or c in status_cols:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            nn = x.notna().mean()
            if nn < 0.3:
                continue
            xv = x.dropna().to_numpy()
            var = float(np.var(xv)) if len(xv) else 0.0
            score = 0.7 * nn + 0.3 * np.log1p(max(var, 0.0))
            if score > best_flow_score:
                best_flow_score = score
                best_flow = c
        flow_col = best_flow

    logger.info(f"列识别：time={best_time}, flow={flow_col}, status_cols={status_cols}")
    return {"time_col": best_time, "flow_col": flow_col, "status_cols": status_cols}


def _guess_header_row(raw_df: pd.DataFrame) -> int:
    scan_n = min(TIME_PARSE_CANDIDATE_ROWS, len(raw_df))
    best_i, best_score = 0, -1.0
    for i in range(scan_n):
        row = raw_df.iloc[i].astype(str).str.strip().fillna("")
        non_empty = (row != "").mean()
        key_hit = sum(any(k in v.lower() for k in TIME_KEYWORDS + FLOW_KEYWORDS + STATUS_KEYWORDS) for v in row)
        score = non_empty + 0.5 * key_hit
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def _parse_mixed_timestamp(series: pd.Series) -> pd.Series:
    """
    混合时间解析（重点修复 Excel 序列数值被误解析为1970附近时间的问题）：
    - 非数值：按常规 datetime 解析；
    - 数值：优先按 Excel 序列日期解析（天，origin=1899-12-30）。
    """
    s = series.copy()
    # 若列本身已是 datetimelike，直接解析返回，避免被转成数值后误处理
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    is_dt_obj = s.apply(lambda x: isinstance(x, (pd.Timestamp, np.datetime64, datetime)))
    numeric = pd.to_numeric(s.where(~is_dt_obj, np.nan), errors="coerce")
    is_num = numeric.notna()

    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    # datetime对象直接保留
    if is_dt_obj.any():
        parsed.loc[is_dt_obj] = pd.to_datetime(s.loc[is_dt_obj], errors="coerce")
    # 非数值文本常规解析
    txt_mask = (~is_num) & (~is_dt_obj)
    if txt_mask.any():
        parsed.loc[txt_mask] = pd.to_datetime(s.loc[txt_mask], errors="coerce")
    # 数值部分按 Excel 序列日期解析（避免 pandas 默认把数字当 ns 时间戳）
    if is_num.any():
        parsed.loc[is_num] = pd.to_datetime(numeric.loc[is_num], unit="D", origin="1899-12-30", errors="coerce")

    return parsed


def load_and_prepare_file(file_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    sheet = xls.sheet_names[0]
    raw = pd.read_excel(file_path, sheet_name=sheet, header=None, engine="openpyxl")
    header_row = _guess_header_row(raw)

    df = pd.read_excel(file_path, sheet_name=sheet, header=header_row, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    col_info = detect_columns(df, logger)

    time_col, flow_col, status_cols = col_info["time_col"], col_info["flow_col"], col_info["status_cols"]
    if time_col is None or flow_col is None:
        raise ValueError("无法识别时间列或主流量列")

    out = pd.DataFrame()
    out["original_row_id"] = np.arange(1, len(df) + 1)
    out["timestamp_raw"] = df[time_col]
    out["timestamp"] = _parse_mixed_timestamp(df[time_col])
    out["raw_value"] = pd.to_numeric(df[flow_col], errors="coerce")
    for c in status_cols:
        out[c] = df[c]

    raw_issue_df = out.copy()
    raw_issue_df["source_file"] = file_path.name
    raw_issue_df["is_time_issue"] = raw_issue_df["timestamp"].isna()
    ts_ok = raw_issue_df["timestamp"].notna()
    raw_issue_df["is_offgrid_timestamp"] = False
    raw_issue_df.loc[ts_ok, "is_offgrid_timestamp"] = (
        (raw_issue_df.loc[ts_ok, "timestamp"].dt.minute % 5 != 0)
        | (raw_issue_df.loc[ts_ok, "timestamp"].dt.second != 0)
        | (raw_issue_df.loc[ts_ok, "timestamp"].dt.microsecond != 0)
    )
    raw_issue_df["is_duplicate_timestamp"] = raw_issue_df["timestamp"].duplicated(keep=False) & raw_issue_df["timestamp"].notna()

    meta = {
        "sheet_name": sheet,
        "header_row": header_row,
        "time_col": time_col,
        "flow_col": flow_col,
        "status_cols": status_cols,
        "raw_rows": len(df),
    }
    logger.info(f"读取 {file_path.name}: sheet={sheet}, header_row={header_row}, rows={len(df)}")
    return out, meta, raw_issue_df


def align_to_5min_grid(df: pd.DataFrame, status_cols: List[str]) -> pd.DataFrame:
    work = df.copy()
    work["is_time_issue"] = work["timestamp"].isna()
    ts_ok = work["timestamp"].notna()
    work["is_offgrid_timestamp"] = False
    work.loc[ts_ok, "is_offgrid_timestamp"] = (
        (work.loc[ts_ok, "timestamp"].dt.minute % 5 != 0)
        | (work.loc[ts_ok, "timestamp"].dt.second != 0)
        | (work.loc[ts_ok, "timestamp"].dt.microsecond != 0)
    )
    work["timestamp_5min"] = pd.NaT
    work.loc[ts_ok, "timestamp_5min"] = work.loc[ts_ok, "timestamp"].dt.floor(SAMPLE_INTERVAL)
    work["is_duplicate_timestamp"] = work["timestamp_5min"].duplicated(keep=False) & work["timestamp_5min"].notna()
    has_disorder = work["timestamp_5min"].dropna().is_monotonic_increasing is False

    work["source_row_indices"] = work["original_row_id"].astype(str)
    agg_map = {
        "raw_value": "mean",
        "is_time_issue": "max",
        "is_offgrid_timestamp": "max",
        "is_duplicate_timestamp": "max",
        "timestamp_raw": lambda x: "|".join([str(v) for v in x if pd.notna(v)]),
        "source_row_indices": lambda x: ",".join(sorted(set([str(v) for v in x if pd.notna(v)]), key=lambda z: int(z)))
    }
    for c in status_cols:
        agg_map[c] = "first"

    grp = work.dropna(subset=["timestamp_5min"]).groupby("timestamp_5min", as_index=False).agg(agg_map)
    grp = grp.sort_values("timestamp_5min").reset_index(drop=True).rename(columns={"timestamp_5min": "timestamp"})

    if len(grp) > 0:
        full_idx = pd.date_range(grp["timestamp"].min(), grp["timestamp"].max(), freq=SAMPLE_INTERVAL)
        aligned = grp.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()
    else:
        aligned = pd.DataFrame(columns=["timestamp", "raw_value", "timestamp_raw", "source_row_indices", "is_time_issue", "is_offgrid_timestamp", "is_duplicate_timestamp"] + status_cols)

    aligned["is_missing"] = aligned["raw_value"].isna()
    aligned["is_disorder"] = has_disorder
    # 统一确保 timestamp 为 datetimelike，避免后续 .dt 报错
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], errors="coerce")
    for c in ["is_time_issue", "is_duplicate_timestamp", "is_offgrid_timestamp"]:
        aligned[c] = aligned.get(c, False).fillna(False).astype(bool)
    aligned["source_row_indices"] = aligned.get("source_row_indices", "").fillna("")
    return aligned


def _status_issue_mask(df: pd.DataFrame, status_cols: List[str]) -> pd.Series:
    if not status_cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    bad_tokens = {"bad", "error", "fault", "异常", "错误", "无效", "fail"}
    for c in status_cols:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.6:
            in_01 = ((num.dropna() >= 0) & (num.dropna() <= 1)).mean() > 0.9
            if in_01:
                mask |= (num <= 0)
            if ("错误率" in c) or ("error" in c.lower()):
                mask |= (num > 0.2)
        else:
            low = s.astype(str).str.lower()
            mask |= low.apply(lambda x: any(t in x for t in bad_tokens))
    return mask.fillna(False)


def build_daily_baseline(df: pd.DataFrame, valid_mask: pd.Series) -> Tuple[pd.Series, pd.Series]:
    tmp = df.copy()
    # 容错：若 timestamp 非 datetimelike，强制转换
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    if tmp["timestamp"].notna().sum() == 0:
        # 极端情况下无可用时间戳，返回常量模板，避免流程崩溃
        g = pd.to_numeric(tmp["raw_value"], errors="coerce").median()
        if pd.isna(g):
            g = 0.0
        base_template = pd.Series(np.repeat(g, 288))
        point_baseline = pd.Series(np.repeat(g, len(tmp)), index=tmp.index)
        return base_template, point_baseline
    tmp["slot"] = tmp["timestamp"].dt.hour * 60 // 5 + tmp["timestamp"].dt.minute // 5
    use_mask = valid_mask.fillna(False)
    if use_mask.sum() == 0:
        use_mask = tmp["raw_value"].notna()
    base_template = tmp.loc[use_mask].groupby("slot")["raw_value"].median()
    full_slots = pd.Series(np.arange(288), name="slot")
    base_template = full_slots.to_frame().merge(base_template.rename("baseline"), left_on="slot", right_index=True, how="left")["baseline"]
    base_template = base_template.interpolate(limit_direction="both")
    if base_template.notna().sum() == 0:
        g = tmp["raw_value"].median()
        if pd.isna(g):
            g = 0.0
        base_template = pd.Series(np.repeat(g, 288))
    point_baseline = tmp["slot"].map(dict(enumerate(base_template.to_numpy())))
    return base_template, point_baseline


def estimate_drift(df: pd.DataFrame, initial_baseline: pd.Series, logger: logging.Logger) -> pd.Series:
    if initial_baseline.notna().sum() == 0:
        logger.warning("初始基线全为空，漂移估计回退为0")
        return pd.Series(0.0, index=df.index)
    resid0 = df["raw_value"] - initial_baseline
    x = np.arange(len(df))
    if HAS_LOWESS:
        good = resid0.notna()
        if good.sum() >= 10:
            drift = pd.Series(np.nan, index=df.index)
            sm = lowess(endog=resid0[good].to_numpy(), exog=x[good], frac=LOWESS_FRAC, return_sorted=False)
            drift.loc[good] = sm
            return drift.interpolate(limit_direction="both")
    return resid0.rolling(DRIFT_ROLLING, center=True, min_periods=max(12, DRIFT_ROLLING // 10)).median().interpolate(limit_direction="both")


def compute_residual(df: pd.DataFrame) -> pd.Series:
    return df["raw_value"] - df["baseline"] - df["drift_raw"]


def detect_spikes(df: pd.DataFrame) -> pd.Series:
    r = df["residual"]
    valid = r.notna()
    if valid.sum() == 0:
        return pd.Series(False, index=df.index)
    d1 = r.diff().abs()
    r_med = r.loc[valid].median()
    r_mad = robust_mad(r.loc[valid])
    d_mad = robust_mad(d1.loc[d1.notna()])
    return (((r - r_med).abs() > SPIKE_MAD_K * 1.4826 * r_mad) & (d1 > DIFF_MAD_K * 1.4826 * d_mad)).fillna(False)


def detect_jumps(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.Series, int]:
    r = df["residual"].copy()
    is_jump = pd.Series(False, index=df.index)
    mad = robust_mad(r)
    thr = max(JUMP_MIN_ABS, JUMP_K * 1.4826 * mad)

    def _select_sparse_peaks(cands: List[int], min_dist: int) -> List[int]:
        """从候选跳变点中做稀疏化，避免几乎每个点都被标记为jump。"""
        if not cands:
            return []
        selected = [cands[0]]
        for p in cands[1:]:
            if p - selected[-1] >= min_dist:
                selected.append(p)
        return selected

    if HAS_RUPTURES and r.notna().sum() > 20:
        series = r.interpolate(limit_direction="both").to_numpy().reshape(-1, 1)
        pen = max(thr, JUMP_MIN_ABS)
        try:
            algo = rpt.Pelt(model="l2").fit(series)
            bkps_raw = [b for b in algo.predict(pen=max(pen, 1e-3)) if b < len(r)]
            cands = []
            for b in bkps_raw:
                left = r.iloc[max(0, b - JUMP_WINDOW):b]
                right = r.iloc[b:min(len(r), b + JUMP_WINDOW)]
                if left.notna().sum() < 2 or right.notna().sum() < 2:
                    continue
                if abs(right.median() - left.median()) >= thr:
                    cands.append(b)
            cands = _select_sparse_peaks(sorted(cands), JUMP_MIN_DISTANCE)
            for b in cands:
                is_jump.iloc[max(0, b - JUMP_MARK_HALF_WIDTH):min(len(r), b + JUMP_MARK_HALF_WIDTH + 1)] = True
            return is_jump, len(cands)
        except Exception as e:
            logger.warning(f"ruptures失败，回退窗口法: {e}")

    cands = []
    for i in range(JUMP_WINDOW, len(r) - JUMP_WINDOW):
        left_win = r.iloc[i - JUMP_WINDOW:i]
        right_win = r.iloc[i:i + JUMP_WINDOW]
        if left_win.notna().sum() < max(2, JUMP_WINDOW // 2) or right_win.notna().sum() < max(2, JUMP_WINDOW // 2):
            continue
        jump_amp = abs(right_win.median() - left_win.median())
        if jump_amp > thr:
            # 局部极值判定：只保留附近窗口内跳幅最大的点，减少密集误报
            local = []
            for j in range(max(JUMP_WINDOW, i - 3), min(len(r) - JUMP_WINDOW, i + 4)):
                l = r.iloc[j - JUMP_WINDOW:j]
                rr = r.iloc[j:j + JUMP_WINDOW]
                if l.notna().sum() >= 2 and rr.notna().sum() >= 2:
                    local.append(abs(rr.median() - l.median()))
            if local and jump_amp >= max(local):
                cands.append(i)

    cands = _select_sparse_peaks(sorted(cands), JUMP_MIN_DISTANCE)
    for i in cands:
        is_jump.iloc[max(0, i - JUMP_MARK_HALF_WIDTH):min(len(r), i + JUMP_MARK_HALF_WIDTH + 1)] = True
    return is_jump, len(cands)


def detect_stuck(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    x = df["raw_value"]
    mask = pd.Series(False, index=df.index)
    for i in range(0, len(x) - STUCK_WINDOW + 1):
        w = x.iloc[i:i + STUCK_WINDOW]
        if w.isna().mean() > 0.2:
            continue
        wv = w.dropna().to_numpy()
        if len(wv) < STUCK_WINDOW - 1:
            continue
        var_ok = np.var(wv) <= STUCK_VAR_THR
        diff_ok = np.max(np.abs(np.diff(wv))) <= STUCK_MAX_DIFF_THR if len(wv) >= 2 else False
        uniq_ok = len(np.unique(np.round(wv, 6))) <= STUCK_UNIQUE_MAX
        if var_ok and ((diff_ok and uniq_ok) or (diff_ok and np.var(wv) < STUCK_VAR_THR * 0.1) or (uniq_ok and np.var(wv) < STUCK_VAR_THR * 0.1)):
            mask.iloc[i:i + STUCK_WINDOW] = True
    return mask, len(contiguous_segments(mask))


def compute_anchored_drift(relative_slow_trend: pd.Series, timestamps: pd.Series) -> Tuple[pd.Series, float]:
    """
    anchored_drift 计算：
    - relative_slow_trend：相对整月baseline的低频慢变项（可跨0）；
    - anchored_drift：以文件开头参考段中位数为锚点后的偏移量，更接近“从初始状态出发的漂移”。
    """
    ts = pd.to_datetime(timestamps, errors="coerce")
    rs = relative_slow_trend.copy()
    valid = ts.notna() & rs.notna()
    if valid.sum() == 0:
        return pd.Series(0.0, index=relative_slow_trend.index), 0.0

    t0 = ts[valid].min()
    ref_end = t0 + pd.Timedelta(days=ANCHOR_REF_DAYS)
    ref_mask = valid & (ts <= ref_end)
    if ref_mask.sum() < 24:
        ref_mask = valid.iloc[: min(288, valid.sum())]

    ref_val = float(rs.loc[ref_mask].median()) if ref_mask.sum() else float(rs[valid].median())
    anchored = rs - ref_val
    return anchored, ref_val


def classify_drift_segments(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """将 drift_raw 分解为 gradual_drift 与 level_shift，并输出段级汇总。"""
    out = df.copy()
    # 兼容旧字段：drift_raw == relative_slow_trend
    drift = out["relative_slow_trend"].fillna(0.0)
    drift_smooth = drift.rolling(12, center=True, min_periods=3).median().interpolate(limit_direction="both")

    drift_dev = (drift_smooth - drift_smooth.rolling(288, center=True, min_periods=24).median()).abs()
    thr = max(DRIFT_ABS_MIN * 0.8, DRIFT_SEGMENT_MAD_K * 1.4826 * robust_mad(drift_smooth))
    cand = (drift_smooth.abs() > thr) | (drift_dev > thr * 0.5)
    segs = merge_segments(contiguous_segments(cand), DRIFT_SEGMENT_MERGE_GAP)
    diff_mad_global = robust_mad(drift_smooth.diff().dropna())

    out["gradual_drift"] = 0.0
    out["level_shift"] = 0.0
    out["segment_id"] = ""
    out["drift_segment_type"] = "none"
    out["is_gradual_drift"] = False
    out["is_level_shift"] = False

    stuck_guard = pd.Series(False, index=out.index)
    if "is_stuck" in out.columns:
        for s0, e0 in contiguous_segments(out["is_stuck"].fillna(False)):
            gs = max(0, s0 - STUCK_GUARD_POINTS)
            ge = min(len(out) - 1, e0 + STUCK_GUARD_POINTS)
            stuck_guard.iloc[gs:ge + 1] = True

    rows = []
    sid = 1
    for s, e in segs:
        seg = out.iloc[s:e + 1]
        n = e - s + 1
        if n < min(PLATEAU_MIN_LEN, GRADUAL_MIN_LEN):
            continue

        seg_smooth = drift_smooth.iloc[s:e + 1]
        pre = drift_smooth.iloc[max(0, s - EDGE_WINDOW):s]
        post = drift_smooth.iloc[e + 1:min(len(out), e + 1 + EDGE_WINDOW)]
        pre_med = pre.median() if pre.notna().sum() else 0.0
        post_med = post.median() if post.notna().sum() else 0.0
        seg_med = seg_smooth.median()

        bg = np.nanmedian([pre_med, post_med])
        amp = float(seg_med - bg)
        left_jump = float(seg_smooth.iloc[0] - pre_med) if len(seg_smooth) else 0.0
        right_jump = float(post_med - seg_smooth.iloc[-1]) if len(seg_smooth) and post.notna().sum() else 0.0

        y = seg_smooth.fillna(seg_med).to_numpy()
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
        inner_std = float(np.nanstd(y))
        # 平顶比例：段内部的一阶导较小点占比
        dy = np.abs(np.diff(y)) if len(y) > 1 else np.array([0.0])
        flat_thr = max(1e-6, np.nanmedian(dy) * 0.8)
        flatness_ratio = float((dy <= flat_thr).mean()) if len(dy) else 1.0
        edge_strength = abs(left_jump) + abs(right_jump)
        mid_change = float(np.nanmean(dy)) if len(dy) else 0.0
        edge_sharpness_ratio = edge_strength / max(mid_change, 1e-6)
        seg_score = 0.5 * flatness_ratio + 0.5 * min(edge_sharpness_ratio / 2.5, 1.0)

        stuck_ratio = float(seg["is_stuck"].mean()) if "is_stuck" in seg else 0.0

        plateau_like = (
            n >= PLATEAU_MIN_LEN
            and abs(amp) >= max(DRIFT_ABS_MIN, PLATEAU_MIN_AMPLITUDE)
            and abs(slope) <= PLATEAU_SLOPE_THR
            and flatness_ratio >= PLATEAU_FLATNESS_THR
            and edge_sharpness_ratio >= PLATEAU_EDGE_RATIO_THR
            and abs(left_jump) >= PLATEAU_EDGE_JUMP_K * max(abs(amp) * 0.2, 1e-6)
            and abs(right_jump) >= PLATEAU_EDGE_JUMP_K * max(abs(amp) * 0.2, 1e-6)
        )
        step_like = (
            n >= PLATEAU_MIN_LEN
            and abs(amp) >= max(DRIFT_ABS_MIN, PLATEAU_MIN_AMPLITUDE)
            and abs(left_jump) >= max(abs(amp) * 0.30, 3.0)
            and abs(right_jump) >= max(abs(amp) * 0.30, 3.0)
            and np.nanstd(np.diff(y)) <= max(2.5 * 1.4826 * diff_mad_global, 1.0)
        )
        plateau_like = plateau_like or step_like
        if STUCK_SUPPRESS_DRIFT and stuck_ratio > 0.6:
            seg_type = "none"
        elif plateau_like:
            seg_type = "level_shift"
        else:
            seg_type = "none"

        if plateau_like:
            idx_seg = out.index[s:e + 1]
            idx_level = idx_seg[~stuck_guard.iloc[s:e + 1].to_numpy()]
            out.loc[idx_level, "level_shift"] = drift_smooth.loc[idx_level].fillna(0.0).to_numpy()
            out.loc[idx_level, "is_level_shift"] = True
            out.loc[idx_level, "drift_segment_type"] = "level_shift"
            out.loc[idx_seg[stuck_guard.iloc[s:e + 1].to_numpy()], "drift_segment_type"] = "none"
        else:
            out.loc[out.index[s:e + 1], "drift_segment_type"] = "none"

        seg_id = f"SEG_{sid:03d}"
        sid += 1
        out.loc[out.index[s:e + 1], "segment_id"] = seg_id

        status_ratio = float(out.loc[out.index[s:e + 1], "is_status_issue"].mean()) if "is_status_issue" in out else 0.0
        is_phy = (seg_type == "level_shift") and (status_ratio < 0.2) and (not out.loc[out.index[s:e + 1], "is_time_issue"].any())
        is_fault = False

        rows.append({
            "segment_id": seg_id,
            "segment_type": seg_type,
            "start_time": out.loc[out.index[s], "timestamp"],
            "end_time": out.loc[out.index[e], "timestamp"],
            "duration_points": n,
            "duration_hours": n * 5 / 60,
            "segment_median": seg_med,
            "pre_median": pre_med,
            "post_median": post_med,
            "amplitude": amp,
            "left_edge_jump": left_jump,
            "right_edge_jump": right_jump,
            "inner_slope": slope,
            "inner_std": inner_std,
            "flatness_ratio": flatness_ratio,
            "edge_sharpness_ratio": edge_sharpness_ratio,
            "segment_class_score": seg_score,
            "drift_segment_type": seg_type,
            "status_issue_ratio": status_ratio,
            "is_candidate_sensor_fault": is_fault,
            "is_candidate_physical_event": is_phy,
        })

    # 基于 anchored_like（慢变项）拆分：anchored_like = level_shift + gradual_drift + fluctuation
    anchor_ref = float(drift_smooth.iloc[: min(288, len(drift_smooth))].median()) if len(drift_smooth) else 0.0
    anchored_like = drift_smooth - anchor_ref
    residual_for_level = anchored_like.copy()

    # --- level_shift: 基于“快速边沿 + 平台 + 回落” ---
    edge_w = max(4, LEVEL_EDGE_WIN)
    edge_signal = (
        residual_for_level.rolling(edge_w, center=False, min_periods=max(3, edge_w // 2)).median().shift(-edge_w // 2)
        - residual_for_level.rolling(edge_w, center=False, min_periods=max(3, edge_w // 2)).median().shift(edge_w // 2)
    ).fillna(0.0)
    edge_thr = max(PLATEAU_MIN_AMPLITUDE * 0.6, 2.5 * 1.4826 * robust_mad(edge_signal))
    edge_idx = [i for i in np.where(edge_signal.abs().to_numpy() >= edge_thr)[0] if not stuck_guard.iloc[i]]
    used = pd.Series(False, index=out.index)
    level_segments: List[Tuple[int, int, float, float, float]] = []
    for i in edge_idx:
        if i >= len(out) - PLATEAU_MIN_LEN:
            continue
        left_jump = float(edge_signal.iloc[i])
        sign = 1.0 if left_jump > 0 else -1.0
        j_best, score_best = None, -np.inf
        for j in edge_idx:
            if j <= i + PLATEAU_MIN_LEN:
                continue
            if j - i > 6 * 288:
                break
            right_jump = float(edge_signal.iloc[j])
            if np.sign(right_jump) != -np.sign(left_jump):
                continue
            pre = residual_for_level.iloc[max(0, i - 24):i]
            mid = residual_for_level.iloc[i + 1:j]
            post = residual_for_level.iloc[j + 1:min(len(out), j + 25)]
            if pre.notna().sum() < 6 or mid.notna().sum() < PLATEAU_MIN_LEN or post.notna().sum() < 6:
                continue
            pre_med = float(pre.median())
            mid_med = float(mid.median())
            post_med = float(post.median())
            amp = mid_med - 0.5 * (pre_med + post_med)
            if np.sign(amp) != np.sign(sign):
                continue
            if abs(amp) < max(PLATEAU_MIN_AMPLITUDE, DRIFT_ABS_MIN):
                continue
            ret_tol = max(abs(amp) * LEVEL_RETURN_TOL_RATIO, DRIFT_ABS_MIN * 0.8)
            if abs(post_med - pre_med) > ret_tol:
                continue
            mid_slope = float(np.polyfit(np.arange(len(mid)), mid.to_numpy(), 1)[0]) if len(mid) >= 2 else 0.0
            if abs(mid_slope) > PLATEAU_SLOPE_THR * 1.8:
                continue
            score = abs(amp) - 0.1 * abs(mid_slope) * len(mid)
            if score > score_best:
                score_best = score
                j_best = j
        if j_best is None:
            continue
        s2, e2 = i + 1, j_best
        idx2 = out.index[s2:e2 + 1]
        if used.loc[idx2].mean() > 0.5:
            continue
        pre2 = residual_for_level.iloc[max(0, s2 - 24):s2]
        mid2 = residual_for_level.iloc[s2:e2 + 1]
        post2 = residual_for_level.iloc[e2 + 1:min(len(out), e2 + 25)]
        pre_med2 = float(pre2.median()) if pre2.notna().sum() else 0.0
        mid_med2 = float(mid2.median()) if mid2.notna().sum() else 0.0
        post_med2 = float(post2.median()) if post2.notna().sum() else 0.0
        amp2 = float(mid_med2 - 0.5 * (pre_med2 + post_med2))
        level_segments.append((s2, e2, amp2, pre_med2, post_med2))
        used.loc[idx2] = True

    for s2, e2, amp2, pre_med2, post_med2 in level_segments:
        idx2 = out.index[s2:e2 + 1]
        stuck2 = out.loc[idx2, "is_stuck"].fillna(False) if "is_stuck" in out.columns else pd.Series(False, index=idx2)
        guard2 = stuck_guard.iloc[s2:e2 + 1]
        idx2_level = idx2[(~stuck2.to_numpy()) & (~guard2.to_numpy())]
        if len(idx2_level) < PLATEAU_MIN_LEN:
            continue
        out.loc[idx2_level, "level_shift"] = amp2
        out.loc[idx2_level, "is_level_shift"] = True
        out.loc[idx2_level, "drift_segment_type"] = "level_shift"
        out.loc[idx2[stuck2.to_numpy() | guard2.to_numpy()], "drift_segment_type"] = "none"
        seg_id = f"SEG_{sid:03d}"
        sid += 1
        out.loc[idx2_level, "segment_id"] = seg_id
        rows.append({
            "segment_id": seg_id,
            "segment_type": "level_shift",
            "start_time": out.loc[idx2_level[0], "timestamp"],
            "end_time": out.loc[idx2_level[-1], "timestamp"],
            "duration_points": int(len(idx2_level)),
            "duration_hours": int(len(idx2_level)) * 5 / 60,
            "segment_median": float(out.loc[idx2_level, "relative_slow_trend"].median()),
            "pre_median": pre_med2,
            "post_median": post_med2,
            "amplitude": amp2,
            "left_edge_jump": float(edge_signal.iloc[s2 - 1]) if s2 - 1 >= 0 else np.nan,
            "right_edge_jump": float(edge_signal.iloc[e2]) if e2 < len(edge_signal) else np.nan,
            "inner_slope": float(np.polyfit(np.arange(len(idx2_level)), out.loc[idx2_level, "relative_slow_trend"].to_numpy(), 1)[0]) if len(idx2_level) >= 2 else 0.0,
            "inner_std": float(np.nanstd(out.loc[idx2_level, "relative_slow_trend"])),
            "flatness_ratio": np.nan,
            "edge_sharpness_ratio": np.nan,
            "segment_class_score": np.nan,
            "drift_segment_type": "level_shift",
            "status_issue_ratio": float(out.loc[idx2_level, "is_status_issue"].mean()) if "is_status_issue" in out else 0.0,
            "is_candidate_sensor_fault": False,
            "is_candidate_physical_event": True,
        })

    # level段内不再考虑drift
    out.loc[out["is_level_shift"], "gradual_drift"] = 0.0
    out.loc[out["is_level_shift"], "is_gradual_drift"] = False

    # --- gradual_drift: 在去掉level后的anchored_like上做“起点可检测”的单调缓变估计 ---
    residual_for_grad = anchored_like - out["level_shift"]
    grad_base = residual_for_grad.rolling(GRADUAL_LONG_WIN, center=True, min_periods=max(24, GRADUAL_LONG_WIN // 6)).median().interpolate(limit_direction="both")
    valid_grad = grad_base.notna() & (~out["is_level_shift"]) & (~stuck_guard)
    if valid_grad.sum() >= GRADUAL_MIN_LEN:
        best = None
        for s, e in contiguous_segments(valid_grad):
            if e - s + 1 < GRADUAL_MIN_LEN:
                continue
            seg_full = grad_base.iloc[s:e + 1].copy().interpolate(limit_direction="both")
            if seg_full.notna().sum() < GRADUAL_MIN_LEN:
                continue
            edge_n = max(12, min(48, (e - s + 1) // 8))
            start_ref = float(seg_full.iloc[:edge_n].median())
            end_ref = float(seg_full.iloc[-edge_n:].median())
            total_change = float(end_ref - start_ref)
            if abs(total_change) < max(GRADUAL_TOTAL_CHANGE_MIN, DRIFT_ABS_MIN * 2.0):
                continue
            direction = 1.0 if total_change >= 0 else -1.0
            diffs = seg_full.diff().fillna(0.0)
            local_noise = max(1e-6, 1.4826 * robust_mad(diffs))
            mono_tol = GRADUAL_MONO_TOL_K * local_noise
            mono_ratio = float((direction * diffs >= -mono_tol).mean())
            if mono_ratio < GRADUAL_MONO_RATIO_MIN:
                continue
            score = abs(total_change) * mono_ratio * np.log1p(e - s + 1)
            if (best is None) or (score > best[0]):
                best = (score, s, e, direction, start_ref, total_change, seg_full)

        if best is not None:
            _, s, e, direction, start_ref, total_change, seg_full = best
            onset_gate = max(GRADUAL_ONSET_ABS_MIN, abs(total_change) * GRADUAL_ONSET_FRAC)
            dev = direction * (seg_full - start_ref)
            onset_pos = np.where(dev.to_numpy() >= onset_gate)[0]
            if len(onset_pos) == 0:
                s_on = s
            else:
                s_on = s + int(onset_pos[0])
                for b in range(s_on - 1, max(-1, s_on - GRADUAL_ONSET_BACKTRACK) - 1, -1):
                    if (not valid_grad.iloc[b]) or (direction * (grad_base.iloc[b] - start_ref) < onset_gate * 0.35):
                        break
                    s_on = b

            seg_raw = grad_base.iloc[s_on:e + 1].copy().interpolate(limit_direction="both")
            seg_raw = seg_raw - float(seg_raw.iloc[0])
            raw_diff = seg_raw.diff().fillna(0.0)
            max_rate = max(GRADUAL_SLOPE_MIN * 4.0, 0.8)
            if direction > 0:
                clipped = raw_diff.clip(lower=0.0, upper=max_rate)
                seg = np.maximum.accumulate(clipped.cumsum().to_numpy())
            else:
                clipped = raw_diff.clip(lower=-max_rate, upper=0.0)
                seg = np.minimum.accumulate(clipped.cumsum().to_numpy())
            seg = seg - seg[0]

            resid_seg = residual_for_grad.iloc[s_on:e + 1].fillna(0.0).to_numpy()
            if direction > 0:
                seg = np.minimum(seg, np.maximum(resid_seg, 0.0))
            else:
                seg = np.maximum(seg, np.minimum(resid_seg, 0.0))
            seg = pd.Series(seg, index=out.index[s_on:e + 1])

            if abs(float(seg.iloc[-1] - seg.iloc[0])) >= max(GRADUAL_TOTAL_CHANGE_MIN * 0.5, DRIFT_ABS_MIN * 1.5):
                idxg = out.index[s_on:e + 1]
                out.loc[idxg, "gradual_drift"] = seg.to_numpy()
                out.loc[idxg, "is_gradual_drift"] = True
                out.loc[idxg, "drift_segment_type"] = "gradual_drift"
                seg_id = f"SEG_{sid:03d}"
                sid += 1
                out.loc[idxg, "segment_id"] = seg_id
                rows.append({
                    "segment_id": seg_id,
                    "segment_type": "gradual_drift",
                    "start_time": out.loc[idxg[0], "timestamp"],
                    "end_time": out.loc[idxg[-1], "timestamp"],
                    "duration_points": int(len(idxg)),
                    "duration_hours": int(len(idxg)) * 5 / 60,
                    "segment_median": float(out.loc[idxg, "relative_slow_trend"].median()),
                    "pre_median": np.nan,
                    "post_median": np.nan,
                    "amplitude": float(seg.iloc[-1] - seg.iloc[0]),
                    "left_edge_jump": np.nan,
                    "right_edge_jump": np.nan,
                    "inner_slope": float(np.polyfit(np.arange(len(seg)), seg.to_numpy(), 1)[0]) if len(seg) >= 2 else 0.0,
                    "inner_std": float(np.nanstd(seg)),
                    "flatness_ratio": np.nan,
                    "edge_sharpness_ratio": np.nan,
                    "segment_class_score": np.nan,
                    "drift_segment_type": "gradual_drift",
                    "status_issue_ratio": float(out.loc[idxg, "is_status_issue"].mean()) if "is_status_issue" in out else 0.0,
                    "is_candidate_sensor_fault": False,
                    "is_candidate_physical_event": False,
                })

    out["drift_type"] = out["drift_segment_type"]  # 兼容旧字段

    return out, pd.DataFrame(rows)


def build_segment_summary(df: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
    """融合 jump/stuck 连续段到段级表，并返回统一段汇总。"""
    rows = seg_df.copy()
    next_id = len(rows) + 1
    for m, seg_type in [(df["is_jump"], "jump"), (df["is_stuck"], "stuck")]:
        for s, e in contiguous_segments(m):
            sid = f"SEG_{next_id:03d}"
            next_id += 1
            part = df.iloc[s:e + 1]
            rows = pd.concat([rows, pd.DataFrame([{
                "segment_id": sid,
                "segment_type": seg_type,
                "start_time": part["timestamp"].iloc[0],
                "end_time": part["timestamp"].iloc[-1],
                "duration_points": len(part),
                "duration_hours": len(part) * 5 / 60,
                "segment_median": np.nanmedian(part["raw_value"]),
                "pre_median": np.nan,
                "post_median": np.nan,
                "amplitude": np.nan,
                "left_edge_jump": np.nan,
                "right_edge_jump": np.nan,
                "inner_slope": np.nan,
                "inner_std": np.nan,
                "status_issue_ratio": float(part["is_status_issue"].mean()),
                "is_candidate_sensor_fault": bool(part["is_status_issue"].mean() > 0.2),
                "is_candidate_physical_event": bool((seg_type == "jump") and (part["is_status_issue"].mean() < 0.2)),
            }])], ignore_index=True)
    return rows


def build_cleaned_series(df: pd.DataFrame) -> pd.DataFrame:
    """构建三类清洗结果：观测层/分析用/严格版。"""
    out = df.copy()

    obs_fault = out["is_missing"] | out["is_time_issue"] | out["is_duplicate_timestamp"] | out["is_spike"] | out["is_stuck"]
    obs = out["raw_value"].copy()
    obs.loc[obs_fault] = np.nan
    obs = obs.interpolate(limit=SHORT_GAP_MAX, limit_direction="both")

    analysis = out["raw_value"] - out["anchored_drift"].fillna(0.0)

    # stuck点在分析序列中回填“日基线”，并在段两端做平滑过渡，避免连接处突跳
    stuck_mask = out["is_stuck"].fillna(False)
    baseline = out["baseline"]
    analysis.loc[stuck_mask] = baseline.loc[stuck_mask]

    def _nearest_valid_value(series: pd.Series, start: int, step: int) -> float:
        i = start
        n = len(series)
        while 0 <= i < n:
            v = series.iloc[i]
            if pd.notna(v):
                return float(v)
            i += step
        return np.nan

    edge_blend_points = 3
    for s, e in contiguous_segments(stuck_mask):
        seg_len = e - s + 1
        if seg_len <= 0:
            continue
        k = min(edge_blend_points, seg_len)
        left_ref = _nearest_valid_value(analysis, s - 1, -1)
        right_ref = _nearest_valid_value(analysis, e + 1, 1)

        if pd.notna(left_ref):
            for i in range(k):
                idx = s + i
                cur = analysis.iloc[idx]
                if pd.isna(cur):
                    continue
                w = (i + 1) / (k + 1)
                analysis.iloc[idx] = (1 - w) * left_ref + w * cur

        if pd.notna(right_ref):
            for i in range(k):
                idx = e - i
                cur = analysis.iloc[idx]
                if pd.isna(cur):
                    continue
                w = (i + 1) / (k + 1)
                analysis.iloc[idx] = (1 - w) * right_ref + w * cur

    strict = obs.copy()
    strict.loc[out["is_level_shift"]] = np.nan

    out["cleaned_observation_value"] = obs
    out["cleaned_analysis_value"] = analysis
    out["cleaned_strict_value"] = strict
    out["cleaned_value"] = out["cleaned_observation_value"]  # 兼容旧字段
    return out


def collect_raw_issue_rows(raw_issue_df: pd.DataFrame) -> pd.DataFrame:
    mask = raw_issue_df["is_time_issue"] | raw_issue_df["is_duplicate_timestamp"] | raw_issue_df["is_offgrid_timestamp"]
    return raw_issue_df.loc[mask].copy()


def safe_filter(df: pd.DataFrame, column_name: str, expected_value=True) -> pd.DataFrame:
    """
    安全筛选：
    - 列存在：按条件筛选；
    - 列不存在/空表：返回空表（保留原列结构）；
    - 保证不抛 KeyError。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=(df.columns if isinstance(df, pd.DataFrame) else []))
    if column_name not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df[column_name] == expected_value].copy()


def record_failure(
    failures: List[Dict[str, object]],
    source_file: str,
    stage: str,
    err: Exception,
    partial_outputs_written: bool,
    process_status: str,
    validation_passed: bool = False,
    validation_reason: str = "",
    valid_point_count: int = 0,
    aligned_point_count: int = 0,
    time_span_days: float = 0.0,
    abnormal_timestamp_count: int = 0,
) -> None:
    tb = traceback.format_exc()
    tb_short = "\\n".join(tb.strip().splitlines()[-6:])
    failures.append(
        {
            "source_file": source_file,
            "stage": stage,
            "error_type": type(err).__name__,
            "error_message": str(err),
            "traceback_short": tb_short,
            "partial_outputs_written": partial_outputs_written,
            "process_status": process_status,
            "validation_passed": validation_passed,
            "validation_reason": validation_reason,
            "valid_point_count": valid_point_count,
            "aligned_point_count": aligned_point_count,
            "time_span_days": time_span_days,
            "abnormal_timestamp_count": abnormal_timestamp_count,
        }
    )


def write_failure_reports(run_report_dir: Path, failures: List[Dict[str, object]]) -> None:
    run_report_dir.mkdir(parents=True, exist_ok=True)
    fail_csv = run_report_dir / "failed_files.csv"
    fail_txt = run_report_dir / "failed_files.txt"
    cols = [
        "source_file",
        "stage",
        "error_type",
        "error_message",
        "traceback_short",
        "partial_outputs_written",
        "process_status",
        "validation_passed",
        "validation_reason",
        "valid_point_count",
        "aligned_point_count",
        "time_span_days",
        "abnormal_timestamp_count",
    ]
    pd.DataFrame(failures, columns=cols).to_csv(fail_csv, index=False, encoding="utf-8-sig")
    with open(fail_txt, "w", encoding="utf-8") as f:
        if not failures:
            f.write("No failed files.\\n")
        else:
            for i, r in enumerate(failures, 1):
                f.write(f"[{i}] file={r['source_file']}\\n")
                f.write(f" stage={r['stage']}\\n")
                f.write(f" type={r['error_type']}\\n")
                f.write(f" message={r['error_message']}\\n")
                f.write(" traceback:\\n")
                f.write(f"{r['traceback_short']}\\n")
                f.write("-" * 80 + "\\n")


def export_fault_workbook(df: pd.DataFrame, raw_issue_rows: pd.DataFrame, seg_summary: pd.DataFrame, out_xlsx: Path, status_cols: List[str]) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    core_cols = [
        "source_file", "timestamp", "timestamp_raw", "raw_value", "baseline", "relative_slow_trend", "anchored_drift",
        "drift_raw", "gradual_drift", "level_shift",
        "residual", "is_missing", "is_duplicate_timestamp", "is_time_issue", "is_offgrid_timestamp", "is_status_issue",
        "is_spike", "is_jump", "is_stuck", "is_gradual_drift", "is_level_shift", "segment_id", "fault_type", "drift_segment_type",
        "cleaned_observation_value", "cleaned_analysis_value", "cleaned_strict_value", "source_row_indices"
    ]
    cols = [c for c in core_cols if c in df.columns] + [c for c in status_cols if c in df.columns and c not in core_cols]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        safe_filter(df, "is_spike", True).reindex(columns=cols).to_excel(writer, sheet_name="spike_points", index=False)
        safe_filter(df, "is_stuck", True).reindex(columns=cols).to_excel(writer, sheet_name="stuck_points", index=False)
        safe_filter(df, "is_jump", True).reindex(columns=cols).to_excel(writer, sheet_name="jump_points", index=False)
        safe_filter(df, "is_gradual_drift", True).reindex(columns=cols).to_excel(writer, sheet_name="gradual_drift_points", index=False)
        safe_filter(df, "is_level_shift", True).reindex(columns=cols).to_excel(writer, sheet_name="level_shift_points", index=False)

        issue_mask = pd.Series(False, index=df.index)
        for c in ["is_missing", "is_time_issue", "is_duplicate_timestamp", "is_offgrid_timestamp"]:
            if c in df.columns:
                issue_mask = issue_mask | df[c].fillna(False)
        df.loc[issue_mask].reindex(columns=cols).to_excel(writer, sheet_name="time_or_missing_issues", index=False)

        (seg_summary if isinstance(seg_summary, pd.DataFrame) else pd.DataFrame()).to_excel(
            writer, sheet_name="segment_summary", index=False
        )

        # 关键修复：缺列时不再触发 KeyError: False
        cand_seg = safe_filter(seg_summary if isinstance(seg_summary, pd.DataFrame) else pd.DataFrame(), "is_candidate_physical_event", True)
        cand_points = df.loc[
            safe_filter(df, "is_level_shift", True).index
        ].copy() if "is_level_shift" in df.columns else df.iloc[0:0].copy()
        if "is_status_issue" in cand_points.columns:
            cand_points = cand_points.loc[~cand_points["is_status_issue"].fillna(False)]
        if "is_time_issue" in cand_points.columns:
            cand_points = cand_points.loc[~cand_points["is_time_issue"].fillna(False)]

        if len(cand_points) == 0 and len(cand_seg) > 0:
            # 若点为空但段存在，优先写段摘要
            cand_seg.to_excel(writer, sheet_name="candidate_physical_events", index=False)
        else:
            cand_points.reindex(columns=cols).to_excel(writer, sheet_name="candidate_physical_events", index=False)

        # 原始层问题行，额外保留一张sheet方便追溯（为空也可写）
        (raw_issue_rows if isinstance(raw_issue_rows, pd.DataFrame) else pd.DataFrame()).to_excel(
            writer, sheet_name="raw_issue_rows", index=False
        )


def safe_export_fault_workbook(
    df: pd.DataFrame,
    raw_issue_rows: pd.DataFrame,
    seg_summary: pd.DataFrame,
    out_xlsx: Path,
    status_cols: List[str],
) -> Tuple[bool, str]:
    """
    容错导出工作簿：
    - 成功返回 (True, '')
    - 失败返回 (False, 错误信息)
    """
    try:
        export_fault_workbook(df, raw_issue_rows, seg_summary, out_xlsx, status_cols)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def detect_dominant_time_cluster(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    识别主体时间簇：按时间排序后使用“大间隙切分簇”，选择点数最多的簇。
    返回：主体簇掩码、离群时间点数量。
    """
    if "timestamp" not in df.columns or len(df) == 0:
        return pd.Series(False, index=df.index), 0
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    # 关键修复：仅基于“有观测值”的点识别主时间簇，避免对齐补齐出的长空时间轴干扰
    signal_mask = df["raw_value"].notna() if "raw_value" in df.columns else pd.Series(True, index=df.index)
    valid = ts.notna() & signal_mask.fillna(False)
    if valid.sum() == 0:
        # 回退：若没有观测值，则退化为所有有效时间戳
        valid = ts.notna()
        if valid.sum() == 0:
            return pd.Series(False, index=df.index), len(df)

    order = ts[valid].sort_values().index
    ts_sorted = ts.loc[order]
    gaps = ts_sorted.diff().gt(pd.Timedelta(days=3)).fillna(False)
    cluster_id = gaps.cumsum()
    # 兼顾“点数最多”和“时间跨度足够长”，避免选到窄小簇导致图像只剩一条竖线
    best_cluster, best_score = None, -1.0
    for cid in cluster_id.unique():
        idx = order[cluster_id == cid]
        tseg = ts.loc[idx]
        span_days = float((tseg.max() - tseg.min()) / pd.Timedelta(days=1)) if len(tseg) else 0.0
        score = len(idx) * 0.7 + span_days * 288 * 0.3
        if score > best_score:
            best_score = score
            best_cluster = cid

    major_idx = order[cluster_id == best_cluster]

    mask = pd.Series(False, index=df.index)
    mask.loc[major_idx] = True
    abnormal_count = int(valid.sum() - (mask & valid).sum())
    return mask, abnormal_count


def validate_time_axis(df: pd.DataFrame) -> Dict[str, object]:
    """校验时间轴是否合理，避免少量离群点把横轴拉爆。"""
    res = {
        "passed": True,
        "reason": "",
        "abnormal_timestamp_count": 0,
        "time_span_days": 0.0,
        "dominant_mask": pd.Series(False, index=df.index),
    }
    if len(df) == 0 or "timestamp" not in df.columns:
        res.update({"passed": False, "reason": "empty_or_no_timestamp"})
        return res

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    valid = ts.notna()
    if valid.sum() < 10:
        res.update({"passed": False, "reason": "too_few_valid_timestamps"})
        return res

    dom_mask, abnormal_count = detect_dominant_time_cluster(df)
    dom_ts = ts[dom_mask & valid]
    if dom_ts.empty:
        res.update({"passed": False, "reason": "no_dominant_time_cluster"})
        return res

    span_days = float((dom_ts.max() - dom_ts.min()) / pd.Timedelta(days=1))
    res.update(
        {
            "abnormal_timestamp_count": int(abnormal_count),
            "time_span_days": span_days,
            "dominant_mask": dom_mask,
        }
    )

    dom_year_med = int(dom_ts.dt.year.median())
    if dom_year_med < 2005 or dom_year_med > 2100:
        res.update({"passed": False, "reason": "dominant_year_out_of_range"})
    elif abnormal_count > max(50, int(0.2 * valid.sum())):
        res.update({"passed": False, "reason": "too_many_abnormal_timestamps"})
    elif span_days < 0.1:
        res.update({"passed": False, "reason": "time_span_too_short"})
    return res


def validate_processed_result(result_df: pd.DataFrame) -> Dict[str, object]:
    """综合校验结果有效性，避免“无报错但结果失真”的假成功。"""
    tval = validate_time_axis(result_df)
    aligned_cnt = int(len(result_df))
    valid_points = int(result_df["raw_value"].notna().sum()) if "raw_value" in result_df.columns else 0
    cleaned_points = int(result_df["cleaned_observation_value"].notna().sum()) if "cleaned_observation_value" in result_df.columns else 0

    passed = tval["passed"] and aligned_cnt >= 100 and valid_points >= 50 and cleaned_points >= 30
    reason_parts = []
    if not tval["passed"]:
        reason_parts.append(str(tval["reason"]))
    if aligned_cnt < 100:
        reason_parts.append("aligned_points_too_few")
    if valid_points < 50:
        reason_parts.append("valid_raw_points_too_few")
    if cleaned_points < 30:
        reason_parts.append("cleaned_points_too_few")

    return {
        "passed": passed,
        "reason": ";".join(reason_parts),
        "valid_point_count": valid_points,
        "aligned_point_count": aligned_cnt,
        "time_span_days": float(tval["time_span_days"]),
        "abnormal_timestamp_count": int(tval["abnormal_timestamp_count"]),
        "dominant_mask": tval["dominant_mask"],
    }


def build_plot_dataframe(df: pd.DataFrame, validation_info: Dict[str, object]) -> pd.DataFrame:
    """构建用于绘图的数据：优先主体时间簇，避免离群时间点破坏横轴。"""
    dom_mask = validation_info.get("dominant_mask", pd.Series(False, index=df.index))
    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    plot_df = plot_df.loc[plot_df["timestamp"].notna()].copy()
    if isinstance(dom_mask, pd.Series) and dom_mask.sum() > 0:
        plot_df = plot_df.loc[dom_mask].copy()

    # 二次兜底：时间分位数截断，防止极端离群点把横轴拉爆
    ts = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    valid = ts.notna()
    before_clip = plot_df.copy()
    if valid.sum() >= 20:
        # 年份鲁棒过滤：围绕主体年份裁剪，防止 199x/210x 等极少数离群点破坏横轴
        year_mode = int(ts[valid].dt.year.mode().iloc[0])
        plot_df = plot_df.loc[(ts.dt.year >= year_mode - 1) & (ts.dt.year <= year_mode + 1)].copy()
        ts = pd.to_datetime(plot_df["timestamp"], errors="coerce")
        valid = ts.notna()
        t_low = ts[valid].quantile(0.01)
        t_high = ts[valid].quantile(0.99)
        plot_df = plot_df.loc[(ts >= t_low) & (ts <= t_high)].copy()
        # 若裁剪过严导致可绘制点太少，回退到裁剪前数据
        if len(plot_df) < 50 and len(before_clip) >= 50:
            plot_df = before_clip

    plot_df = plot_df.sort_values("timestamp")
    # 最后防线：按年份主众数再滤一次，避免极少量异常年份点残留
    ts2 = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    valid2 = ts2.notna()
    if valid2.sum() >= 20:
        ymode2 = int(ts2[valid2].dt.year.mode().iloc[0])
        refined = plot_df.loc[(ts2.dt.year >= ymode2 - 1) & (ts2.dt.year <= ymode2 + 1)].copy()
        if len(refined) >= max(50, int(0.6 * len(plot_df))):
            plot_df = refined

    return plot_df


def safe_make_plots(df: pd.DataFrame, file_stem: str, fig_path: Path, validation_info: Dict[str, object]) -> Tuple[bool, str]:
    try:
        plot_df = build_plot_dataframe(df, validation_info)
        # 用户要求每个样本都出图：若过滤后点太少，则回退到原始对齐数据作图
        if len(plot_df) < 10:
            plot_df = df.copy()
        make_plots(plot_df, file_stem, fig_path)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =========================
# 结果解释与标签工具函数
# =========================
def detect_fault_type(row: pd.Series) -> str:
    if row.get("is_time_issue", False) or row.get("is_missing", False):
        return "time_or_missing"
    if row.get("is_duplicate_timestamp", False):
        return "duplicate"
    if row.get("is_spike", False):
        return "spike"
    if row.get("is_stuck", False):
        return "stuck"
    if row.get("is_jump", False):
        return "jump"
    if row.get("is_level_shift", False):
        return "level_shift"
    if row.get("is_gradual_drift", False):
        return "gradual_drift"
    return "none"


def make_plots(df: pd.DataFrame, file_stem: str, fig_path: Path) -> None:
    # 统一作图前时间净化，防止极端年份把横轴拉爆
    p = df.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], errors="coerce")
    p = p.loc[p["timestamp"].notna()].copy()
    p = p.loc[(p["timestamp"].dt.year >= 2010) & (p["timestamp"].dt.year <= 2035)].copy()
    p = p.sort_values("timestamp")
    if len(p) == 0:
        # 即使无有效数据也输出一张诊断图，便于人工排查
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.set_title(f"{file_stem} Plot Warning")
        ax.text(0.5, 0.5, "No valid timestamp/value after filtering", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(p["timestamp"], p["raw_value"], label="raw", lw=1)
    axes[0].plot(p["timestamp"], p["cleaned_observation_value"], label="cleaned_observation", lw=1)
    axes[0].plot(p["timestamp"], p["cleaned_analysis_value"], label="cleaned_analysis", lw=1)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{file_stem} Raw and Cleaned Series")

    # 第二图恢复 gradual_drift + level_shift 两条主线，anchored_drift 作为辅助虚线
    axes[1].plot(p["timestamp"], p["gradual_drift"], label="gradual_drift", lw=1.1)
    axes[1].plot(p["timestamp"], p["level_shift"], label="level_shift", lw=1.1)
    axes[1].plot(p["timestamp"], p["anchored_drift"], label="anchored_drift", lw=0.9, linestyle="--", alpha=0.7)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Drift Components")

    axes[2].plot(p["timestamp"], p["residual"], label="residual", lw=1)
    axes[2].legend(loc="upper right")
    axes[2].set_title("Residual Series")

    # 仅绘制异常点/段，不绘制背景线；纵轴使用 raw_value，便于定位原始序列中的问题位置
    style = [
        ("is_spike", "spike", "tab:red", 14, 0.9),
        ("is_jump", "jump", "tab:orange", 14, 0.85),
        ("is_stuck", "stuck", "tab:purple", 14, 0.85),
        ("is_gradual_drift", "gradual_drift", "tab:blue", 12, 0.65),
        ("is_level_shift", "level_shift", "tab:green", 12, 0.7),
    ]
    for c, name, color, size, alpha in style:
        axes[3].scatter(p.loc[p[c], "timestamp"], p.loc[p[c], "raw_value"], s=size, alpha=alpha, color=color, label=name)
    axes[3].legend(loc="upper right")
    axes[3].set_title("Anomaly Markers")

    # 强制横轴使用当前净化后的时间范围，避免被隐藏离群点影响
    xmin, xmax = p["timestamp"].min(), p["timestamp"].max()
    for ax in axes:
        ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_outputs(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def summarize_time_range(source_file: str, raw_df: pd.DataFrame, aligned_df: pd.DataFrame, plot_df: pd.DataFrame) -> Dict[str, object]:
    """汇总每个样本的时间范围，用于排查横轴异常。"""
    def _minmax(s: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
        ts = pd.to_datetime(s, errors="coerce")
        ts = ts[ts.notna()]
        if len(ts) == 0:
            return pd.NaT, pd.NaT
        return ts.min(), ts.max()

    raw_min, raw_max = _minmax(raw_df.get("timestamp", pd.Series(dtype="datetime64[ns]")))
    aligned_min, aligned_max = _minmax(aligned_df.get("timestamp", pd.Series(dtype="datetime64[ns]")))
    plot_min, plot_max = _minmax(plot_df.get("timestamp", pd.Series(dtype="datetime64[ns]")))

    def _span_days(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return float((b - a) / pd.Timedelta(days=1))

    def _fmt_ts(t):
        if pd.isna(t):
            return ""
        return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "source_file": source_file,
        "raw_ts_min": _fmt_ts(raw_min),
        "raw_ts_max": _fmt_ts(raw_max),
        "raw_span_days": _span_days(raw_min, raw_max),
        "aligned_ts_min": _fmt_ts(aligned_min),
        "aligned_ts_max": _fmt_ts(aligned_max),
        "aligned_span_days": _span_days(aligned_min, aligned_max),
        "plot_ts_min": _fmt_ts(plot_min),
        "plot_ts_max": _fmt_ts(plot_max),
        "plot_span_days": _span_days(plot_min, plot_max),
        "plot_year_min": int(pd.to_datetime(plot_df.get("timestamp", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna().dt.year.min()) if len(plot_df) else np.nan,
        "plot_year_max": int(pd.to_datetime(plot_df.get("timestamp", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna().dt.year.max()) if len(plot_df) else np.nan,
        "raw_valid_points": int(pd.to_numeric(raw_df.get("raw_value", pd.Series(dtype=float)), errors="coerce").notna().sum()),
        "aligned_valid_points": int(pd.to_numeric(aligned_df.get("raw_value", pd.Series(dtype=float)), errors="coerce").notna().sum()),
        "plot_points": int(len(plot_df)),
    }


# =========================
# 主流程入口
# =========================
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_root = base_dir / "输出" / "第一题流量处理结果"
    per_file_dir = out_root / "per_file_csv"
    fig_dir = out_root / "figures"
    workbook_dir = out_root / "fault_workbooks"
    cleaned_dir = out_root / "cleaned_data"
    report_dir = out_root / "run_reports"

    if CLEAR_PREVIOUS_OUTPUT and out_root.exists():
        shutil.rmtree(out_root)

    for d in [out_root, per_file_dir, fig_dir, workbook_dir, cleaned_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "run.log")
    logger.info("开始处理流量数据批任务")
    logger.info(f"LOWESS可用={HAS_LOWESS}, ruptures可用={HAS_RUPTURES}")

    files = find_input_files(base_dir)
    if not files:
        logger.error("未找到xlsx文件")
        print("共处理0个文件，成功0个，失败0个")
        return

    summary_rows = []
    time_range_rows = []
    ok, partial_failed, invalid_count, fail = 0, 0, 0, 0
    failure_records: List[Dict[str, object]] = []

    for fp in files:
        partial_outputs_written = False
        file_status = "success"
        stage = "init"
        try:
            stage = "load_and_prepare_file"
            raw_df, meta, raw_issue_df = load_and_prepare_file(fp, logger)
            status_cols = meta["status_cols"]

            stage = "align_to_5min_grid"
            aligned = align_to_5min_grid(raw_df, status_cols)
            aligned["source_file"] = fp.name

            stage = "build_daily_baseline"
            aligned["is_status_issue"] = _status_issue_mask(aligned, status_cols)
            aligned["is_observation_anomaly"] = aligned["is_missing"] | aligned["is_duplicate_timestamp"] | aligned["is_time_issue"] | aligned["is_status_issue"]

            valid0 = (~aligned["is_observation_anomaly"]) & aligned["raw_value"].notna()
            _, baseline0 = build_daily_baseline(aligned, valid0)
            aligned["baseline"] = baseline0

            stage = "estimate_relative_slow_trend"
            # drift_raw 为兼容旧字段保留；语义上等价于 relative_slow_trend
            aligned["relative_slow_trend"] = estimate_drift(aligned, aligned["baseline"], logger)
            aligned["drift_raw"] = aligned["relative_slow_trend"]
            aligned["anchored_drift"], reference_anchor_value = compute_anchored_drift(
                aligned["relative_slow_trend"], aligned["timestamp"]
            )

            valid1 = (~aligned["is_observation_anomaly"]) & aligned["raw_value"].notna() & aligned["relative_slow_trend"].notna()
            _, baseline1 = build_daily_baseline(
                aligned.assign(raw_value=aligned["raw_value"] - aligned["relative_slow_trend"]),
                valid1,
            )
            aligned["baseline"] = baseline1

            stage = "detect_spikes"
            aligned["residual"] = compute_residual(aligned)
            aligned["is_spike"] = detect_spikes(aligned)

            stage = "detect_jumps"
            aligned["is_jump"], jump_segments = detect_jumps(aligned, logger)

            stage = "detect_stuck"
            aligned["is_stuck"], stuck_segments = detect_stuck(aligned)

            stage = "classify_drift_segments"
            aligned, drift_seg_df = classify_drift_segments(aligned)
            seg_summary = build_segment_summary(aligned, drift_seg_df)
            if len(seg_summary) > 0:
                seg_summary["reference_anchor_value"] = reference_anchor_value

            stage = "build_cleaned_series"
            aligned["is_candidate_physical_event"] = aligned["is_level_shift"] & (~aligned["is_status_issue"]) & (~aligned["is_time_issue"])
            aligned = build_cleaned_series(aligned)
            aligned["fault_type"] = aligned.apply(detect_fault_type, axis=1)
            validation_info = validate_processed_result(aligned)
            validation_passed = bool(validation_info["passed"])
            validation_reason = str(validation_info["reason"])
            if not validation_passed:
                file_status = "invalid_result"
            plot_df_for_stats = build_plot_dataframe(aligned, validation_info)
            time_range_rows.append(summarize_time_range(fp.name, raw_df, aligned, plot_df_for_stats))

            out_cols_base = [
                "source_file", "timestamp", "timestamp_raw", "source_row_indices", "raw_value", "baseline", "drift_raw",
                "relative_slow_trend", "anchored_drift", "gradual_drift", "level_shift", "drift_type", "drift_segment_type",
                "residual", "is_missing", "is_duplicate_timestamp",
                "is_time_issue", "is_offgrid_timestamp", "is_status_issue", "is_spike", "is_jump", "is_stuck",
                "is_gradual_drift", "is_level_shift", "segment_id", "fault_type", "is_observation_anomaly",
                "is_candidate_physical_event", "cleaned_observation_value", "cleaned_analysis_value", "cleaned_strict_value",
                "cleaned_value"
            ]
            out_cols = out_cols_base + [c for c in status_cols if c in aligned.columns]
            out_df = aligned[out_cols].copy()

            stage = "save_outputs"
            save_outputs(out_df, per_file_dir / f"{fp.stem}_result.csv")
            save_outputs(
                aligned[["timestamp", "raw_value", "cleaned_observation_value", "cleaned_analysis_value", "cleaned_strict_value"]],
                cleaned_dir / f"{fp.stem}_cleaned.csv",
            )
            partial_outputs_written = True

            # 非关键输出阶段：失败记 partial_failed，不计入完全失败
            stage = "make_plots"
            ok_plot, plot_msg = safe_make_plots(aligned, fp.stem, fig_dir / f"{fp.stem}_overview.png", validation_info)
            if not ok_plot:
                if file_status != "invalid_result":
                    file_status = "partial_failed"
                err = RuntimeError(plot_msg)
                record_failure(
                    failure_records, fp.name, "make_plots", err, partial_outputs_written, "partial_failed",
                    validation_passed=validation_passed, validation_reason=validation_reason,
                    valid_point_count=int(validation_info["valid_point_count"]),
                    aligned_point_count=int(validation_info["aligned_point_count"]),
                    time_span_days=float(validation_info["time_span_days"]),
                    abnormal_timestamp_count=int(validation_info["abnormal_timestamp_count"]),
                )
                logger.error(f"部分失败（绘图）: {fp.name}, error={plot_msg}")

            stage = "export_fault_workbook"
            raw_issue_rows = collect_raw_issue_rows(raw_issue_df)
            ok_wb, wb_msg = safe_export_fault_workbook(
                aligned, raw_issue_rows, seg_summary, workbook_dir / f"{fp.stem}_faults.xlsx", status_cols
            )
            if not ok_wb:
                if file_status != "invalid_result":
                    file_status = "partial_failed"
                err = RuntimeError(wb_msg)
                record_failure(
                    failure_records, fp.name, "export_fault_workbook", err, partial_outputs_written, "partial_failed",
                    validation_passed=validation_passed, validation_reason=validation_reason,
                    valid_point_count=int(validation_info["valid_point_count"]),
                    aligned_point_count=int(validation_info["aligned_point_count"]),
                    time_span_days=float(validation_info["time_span_days"]),
                    abnormal_timestamp_count=int(validation_info["abnormal_timestamp_count"]),
                )
                logger.error(f"部分失败（工作簿）: {fp.name}, error={wb_msg}")

            if file_status == "invalid_result":
                err = RuntimeError(f"validation_failed: {validation_reason}")
                record_failure(
                    failure_records, fp.name, "validate_processed_result", err, partial_outputs_written, "invalid_result",
                    validation_passed=validation_passed, validation_reason=validation_reason,
                    valid_point_count=int(validation_info["valid_point_count"]),
                    aligned_point_count=int(validation_info["aligned_point_count"]),
                    time_span_days=float(validation_info["time_span_days"]),
                    abnormal_timestamp_count=int(validation_info["abnormal_timestamp_count"]),
                )

            summary_rows.append({
                "文件名": fp.name,
                "process_status": file_status,
                "validation_passed": validation_passed,
                "validation_reason": validation_reason,
                "valid_point_count": int(validation_info["valid_point_count"]),
                "aligned_point_count": int(validation_info["aligned_point_count"]),
                "time_span_days": float(validation_info["time_span_days"]),
                "abnormal_timestamp_count": int(validation_info["abnormal_timestamp_count"]),
                "原始记录数": meta["raw_rows"],
                "时间对齐后记录数": len(aligned),
                "缺失点数": int(aligned["is_missing"].sum()),
                "重复点数": int(aligned["is_duplicate_timestamp"].sum()),
                "状态异常点数": int(aligned["is_status_issue"].sum()),
                "尖峰点数": int(aligned["is_spike"].sum()),
                "跳变段数": int(jump_segments),
                "stuck段数": int(stuck_segments),
                "gradual_drift段数": int((seg_summary["segment_type"] == "gradual_drift").sum()) if len(seg_summary) else 0,
                "level_shift段数": int((seg_summary["segment_type"] == "level_shift").sum()) if len(seg_summary) else 0,
                "gradual_drift点数": int(aligned["is_gradual_drift"].sum()),
                "level_shift点数": int(aligned["is_level_shift"].sum()),
                "候选物理事件段数": int(seg_summary["is_candidate_physical_event"].sum()) if len(seg_summary) and "is_candidate_physical_event" in seg_summary else 0,
                "观测层清洗后可用点数": int(aligned["cleaned_observation_value"].notna().sum()),
                "严格清洗后可用点数": int(aligned["cleaned_strict_value"].notna().sum()),
                "最终可用点数": int(aligned["cleaned_value"].notna().sum()),
                "自动识别到的时间列名": meta["time_col"],
                "自动识别到的主流量列名": meta["flow_col"],
                "读取工作表": meta["sheet_name"],
            })
            if file_status == "success":
                ok += 1
                logger.info(f"完成：{fp.name}")
            elif file_status == "invalid_result":
                invalid_count += 1
                logger.warning(f"结果无效：{fp.name}")
            else:
                partial_failed += 1
                logger.warning(f"部分失败：{fp.name}")
        except Exception as e:
            fail += 1
            if not any(r.get("source_file") == fp.name for r in time_range_rows):
                time_range_rows.append(
                    {
                        "source_file": fp.name,
                        "raw_ts_min": pd.NaT,
                        "raw_ts_max": pd.NaT,
                        "raw_span_days": np.nan,
                        "aligned_ts_min": pd.NaT,
                        "aligned_ts_max": pd.NaT,
                        "aligned_span_days": np.nan,
                        "plot_ts_min": pd.NaT,
                        "plot_ts_max": pd.NaT,
                        "plot_span_days": np.nan,
                        "raw_valid_points": 0,
                        "aligned_valid_points": 0,
                        "plot_points": 0,
                    }
                )
            record_failure(failure_records, fp.name, stage, e, partial_outputs_written, "failed")
            logger.exception(f"失败：{fp.name}, stage={stage}, error={e}")

    pd.DataFrame(summary_rows).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(time_range_rows).to_csv(report_dir / "sample_time_ranges.csv", index=False, encoding="utf-8-sig")
    write_failure_reports(report_dir, failure_records)
    run_summary = pd.DataFrame(
        [
            {
                "total_files": len(files),
                "success_count": ok,
                "partial_failed_count": partial_failed,
                "invalid_result_count": invalid_count,
                "failed_count": fail,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "output_dir": str(out_root),
            }
        ]
    )
    run_summary.to_csv(report_dir / "run_summary.csv", index=False, encoding="utf-8-sig")

    failed_names = sorted(set([r["source_file"] for r in failure_records if r["process_status"] == "failed"]))
    logger.info(
        f"任务结束：共处理{len(files)}个文件，成功{ok}个，部分失败{partial_failed}个，结果无效{invalid_count}个，完全失败{fail}个"
    )
    print(f"共处理{len(files)}个文件")
    print(f"成功：{ok}")
    print(f"部分失败：{partial_failed}")
    print(f"结果无效：{invalid_count}")
    print(f"完全失败：{fail}")
    if failed_names:
        print(f"失败文件：{', '.join(failed_names)}")
    print(f"失败详情见：{report_dir / 'failed_files.csv'}")


if __name__ == "__main__":
    main()
