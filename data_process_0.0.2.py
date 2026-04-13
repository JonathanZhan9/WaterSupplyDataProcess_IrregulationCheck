#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
供水管网流量数据批处理脚本
处理逻辑：先做时间对齐和观测层质控，再估计日基线和漂移，随后在残差层识别尖峰、跳变和stuck，最后进行保守清洗并输出结果。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 可调参数区（统一管理）
# =========================
SAMPLE_INTERVAL = "5min"  # 题目要求固定 5 分钟
STUCK_WINDOW = 8  # 8 点=40 分钟
SHORT_GAP_MAX = 3  # 短缺失上限，<=3 允许插值
SPIKE_MAD_K = 6.0  # 尖峰残差阈值倍数（稳健）
DIFF_MAD_K = 6.0  # 尖峰一阶差分阈值倍数（稳健）
STUCK_VAR_THR = 1e-3  # stuck 方差阈值（极平坦）
STUCK_MAX_DIFF_THR = 1e-3  # stuck 窗口内最大相邻差分阈值
STUCK_UNIQUE_MAX = 2  # stuck 窗口唯一值个数上限
JUMP_WINDOW = 6  # 跳变替代算法前后窗口长度（6点=30分钟）
JUMP_K = 8.0  # 跳变阈值倍数（基于残差MAD）
LOWESS_FRAC = 0.04  # LOWESS 平滑强度，约对应低频慢变
DRIFT_ROLLING = 288  # 无 LOWESS 时回退滚动窗口（1天）
TIME_PARSE_CANDIDATE_ROWS = 20  # 猜测表头时查看前20行

TIME_KEYWORDS = ["时间", "日期", "datetime", "time", "timestamp"]
FLOW_KEYWORDS = ["流量", "瞬时流量", "流量值", "value", "测量值"]
STATUS_KEYWORDS = ["数据质量", "io错误率", "质量", "错误", "状态", "quality", "error", "status"]


# 软依赖：statsmodels / ruptures（若不存在自动回退）
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
    """创建日志器，输出到终端和文件。"""
    logger = logging.getLogger("flow_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def robust_mad(x: pd.Series | np.ndarray) -> float:
    """稳健 MAD（防止 0 除）。"""
    arr = pd.Series(x).dropna().to_numpy()
    if len(arr) == 0:
        return 1e-12
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(mad, 1e-12)


def contiguous_segments(mask: pd.Series) -> List[Tuple[int, int]]:
    """将布尔序列转成连续区间列表（闭区间索引）。"""
    arr = mask.fillna(False).to_numpy(dtype=bool)
    segs = []
    start = None
    for i, flag in enumerate(arr):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(arr) - 1))
    return segs


def find_input_files(base_dir: Path) -> List[Path]:
    """寻找输入文件，优先题目给定目录，兼容直接放在脚本同级目录。"""
    primary = base_dir / "数据" / "第一题、流量数据"
    if primary.exists():
        files = sorted(primary.glob("*.xlsx"))
        if files:
            return files
    # 兼容示例仓库结构
    return sorted(base_dir.glob("样本*.xlsx"))


def detect_columns(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, object]:
    """自动识别时间列、主流量列和状态列。"""
    cols = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in cols]

    # 时间列候选评分：关键词 + 可解析率 + 非空率
    time_candidates = [c for c in cols if any(k in c.lower() for k in TIME_KEYWORDS)]
    if not time_candidates:
        time_candidates = cols[:]  # 兜底：全部列尝试

    best_time = None
    best_score = -1.0
    for c in time_candidates:
        s = df[c]
        parsed = pd.to_datetime(s, errors="coerce")
        parse_rate = parsed.notna().mean()
        non_null = s.notna().mean()
        score = parse_rate * 0.7 + non_null * 0.3
        if score > best_score:
            best_score = score
            best_time = c

    # 状态列识别
    status_cols = [c for c in cols if any(k in c.lower() for k in STATUS_KEYWORDS)]

    # 主流量列候选
    flow_candidates = [c for c in cols if any(k in c.lower() for k in FLOW_KEYWORDS)]

    if not flow_candidates:
        numeric_cols = []
        for c in cols:
            if c == best_time or c in status_cols:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.3:
                numeric_cols.append(c)
        # 用非空率+方差筛选，避免纯常量列
        best_flow = None
        best_flow_score = -1.0
        for c in numeric_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            nn = x.notna().mean()
            var = float(np.nanvar(x.to_numpy()))
            var_score = np.log1p(max(var, 0.0))
            score = 0.7 * nn + 0.3 * var_score
            if score > best_flow_score:
                best_flow_score = score
                best_flow = c
        flow_col = best_flow
    else:
        # 有关键词时，选非空率最高
        flow_col = max(flow_candidates, key=lambda c: pd.to_numeric(df[c], errors="coerce").notna().mean())

    logger.info(f"列识别：time={best_time}, flow={flow_col}, status_cols={status_cols}")
    return {"time_col": best_time, "flow_col": flow_col, "status_cols": status_cols}


def _guess_header_row(raw_df: pd.DataFrame) -> int:
    """启发式猜测真正表头行，兼容前置空行/说明行。"""
    scan_n = min(TIME_PARSE_CANDIDATE_ROWS, len(raw_df))
    best_i, best_score = 0, -1.0
    for i in range(scan_n):
        row = raw_df.iloc[i].astype(str).str.strip().fillna("")
        non_empty = (row != "").mean()
        key_hit = sum(any(k in v.lower() for k in TIME_KEYWORDS + FLOW_KEYWORDS + STATUS_KEYWORDS) for v in row)
        score = non_empty + key_hit * 0.5
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def _parse_mixed_timestamp(series: pd.Series) -> pd.Series:
    """
    兼容混合时间格式解析：
    1) 先按常规字符串/时间戳解析；
    2) 对仍未解析成功且可转数值的项，按 Excel 序列日期（天）解析。
    这样可避免把整列 datetime 字符串传给 origin+unit 导致 ValueError。
    """
    parsed = pd.to_datetime(series, errors="coerce")
    unresolved = parsed.isna()
    if unresolved.any():
        numeric = pd.to_numeric(series, errors="coerce")
        excel_mask = unresolved & numeric.notna()
        if excel_mask.any():
            parsed.loc[excel_mask] = pd.to_datetime(
                numeric.loc[excel_mask],
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )
    return parsed


def load_and_prepare_file(file_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """读取文件、自动识别表头和关键列，输出标准字段。"""
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    sheet = xls.sheet_names[0]
    raw = pd.read_excel(file_path, sheet_name=sheet, header=None, engine="openpyxl")
    header_row = _guess_header_row(raw)

    df = pd.read_excel(file_path, sheet_name=sheet, header=header_row, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    col_info = detect_columns(df, logger)

    time_col = col_info["time_col"]
    flow_col = col_info["flow_col"]
    status_cols = col_info["status_cols"]

    if time_col is None or flow_col is None:
        raise ValueError("无法识别时间列或主流量列")

    out = pd.DataFrame()
    out["timestamp_raw"] = df[time_col]
    out["timestamp"] = _parse_mixed_timestamp(df[time_col])
    out["raw_value"] = pd.to_numeric(df[flow_col], errors="coerce")

    for c in status_cols:
        out[c] = df[c]

    meta = {
        "sheet_name": sheet,
        "header_row": header_row,
        "time_col": time_col,
        "flow_col": flow_col,
        "status_cols": status_cols,
        "raw_rows": len(df),
    }
    logger.info(f"读取 {file_path.name}: sheet={sheet}, header_row={header_row}, rows={len(df)}")
    return out, meta


def align_to_5min_grid(df: pd.DataFrame, status_cols: List[str]) -> pd.DataFrame:
    """5分钟对齐：先标记重复/乱序，再按时间聚合并补全网格。"""
    work = df.copy()
    work["is_time_issue"] = work["timestamp"].isna()

    work["is_duplicate_timestamp"] = work["timestamp"].duplicated(keep=False) & work["timestamp"].notna()
    work["_order"] = np.arange(len(work))
    has_disorder = work["timestamp"].dropna().is_monotonic_increasing is False

    # 保留重复标记，再做按时间聚合
    agg_map = {"raw_value": "mean", "is_time_issue": "max", "is_duplicate_timestamp": "max", "_order": "min"}
    for c in status_cols:
        agg_map[c] = "first"

    grp = work.dropna(subset=["timestamp"]).groupby("timestamp", as_index=False).agg(agg_map)
    grp = grp.sort_values("timestamp").reset_index(drop=True)

    if len(grp) > 0:
        full_idx = pd.date_range(grp["timestamp"].min().floor(SAMPLE_INTERVAL), grp["timestamp"].max().ceil(SAMPLE_INTERVAL), freq=SAMPLE_INTERVAL)
        aligned = grp.set_index("timestamp").reindex(full_idx).rename_axis("timestamp").reset_index()
    else:
        aligned = pd.DataFrame(columns=["timestamp", "raw_value", "is_time_issue", "is_duplicate_timestamp"] + status_cols)

    aligned["is_missing"] = aligned["raw_value"].isna()
    aligned["is_disorder"] = has_disorder

    for c in ["is_time_issue", "is_duplicate_timestamp"]:
        if c in aligned.columns:
            aligned[c] = aligned[c].fillna(False).astype(bool)

    for c in status_cols:
        if c not in aligned.columns:
            aligned[c] = np.nan

    return aligned


def _status_issue_mask(df: pd.DataFrame, status_cols: List[str]) -> pd.Series:
    """状态异常辅助标记：识别明显坏值（0/False/异常字符串/高错误率）。"""
    if not status_cols:
        return pd.Series(False, index=df.index)

    mask = pd.Series(False, index=df.index)
    bad_tokens = {"bad", "error", "fault", "异常", "错误", "无效", "fail"}

    for c in status_cols:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        # 数值状态列：如果是[0,1]风格，则0视作坏；错误率列高于0.2视作异常
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
    """按一天288个时刻做中位数基线，返回模板和逐点基线。"""
    tmp = df.copy()
    tmp["slot"] = tmp["timestamp"].dt.hour * 60 // 5 + tmp["timestamp"].dt.minute // 5

    base_template = tmp.loc[valid_mask].groupby("slot")["raw_value"].median()
    full_slots = pd.Series(np.arange(288), name="slot")
    base_template = full_slots.to_frame().merge(base_template.rename("baseline"), left_on="slot", right_index=True, how="left")["baseline"]
    base_template = base_template.interpolate(limit_direction="both")

    point_baseline = tmp["slot"].map(dict(enumerate(base_template.to_numpy())))
    return base_template, point_baseline


def estimate_drift(df: pd.DataFrame, initial_baseline: pd.Series, logger: logging.Logger) -> pd.Series:
    """估计慢变漂移：优先 LOWESS，缺包时退化到滚动中位数。"""
    resid0 = df["raw_value"] - initial_baseline
    x = np.arange(len(df))

    if HAS_LOWESS:
        good = resid0.notna()
        if good.sum() < 10:
            logger.warning("LOWESS 可用，但有效点太少，改用滚动中位数")
        else:
            drift = pd.Series(np.nan, index=df.index)
            sm = lowess(endog=resid0[good].to_numpy(), exog=x[good], frac=LOWESS_FRAC, return_sorted=False)
            drift.loc[good] = sm
            drift = drift.interpolate(limit_direction="both")
            logger.info("漂移估计：使用 LOWESS")
            return drift

    logger.info("漂移估计：使用滚动中位数回退方案")
    return resid0.rolling(DRIFT_ROLLING, center=True, min_periods=max(12, DRIFT_ROLLING // 10)).median().interpolate(limit_direction="both")


def compute_residual(df: pd.DataFrame) -> pd.Series:
    """残差=原始值-基线-漂移。"""
    return df["raw_value"] - df["baseline"] - df["drift"]


def detect_spikes(df: pd.DataFrame) -> pd.Series:
    """尖峰检测：残差幅值+一阶差分联合稳健阈值。"""
    r = df["residual"]
    d1 = r.diff().abs()
    r_med = np.nanmedian(r)
    r_mad = robust_mad(r)
    d_mad = robust_mad(d1)

    cond_r = (r - r_med).abs() > SPIKE_MAD_K * 1.4826 * r_mad
    cond_d = d1 > DIFF_MAD_K * 1.4826 * d_mad
    return (cond_r & cond_d).fillna(False)


def detect_jumps(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.Series, int]:
    """跳变检测：优先 ruptures.PELT，缺包时改用前后窗中位数差。"""
    r = df["residual"].copy()
    is_jump = pd.Series(False, index=df.index)

    if HAS_RUPTURES and r.notna().sum() > 20:
        series = r.interpolate(limit_direction="both").to_numpy().reshape(-1, 1)
        pen = JUMP_K * 1.4826 * robust_mad(r)
        try:
            algo = rpt.Pelt(model="l2").fit(series)
            bkps = algo.predict(pen=max(pen, 1e-3))
            bkps = [b for b in bkps if b < len(r)]
            for b in bkps:
                left = max(0, b - JUMP_WINDOW)
                right = min(len(r) - 1, b + JUMP_WINDOW)
                is_jump.iloc[left : right + 1] = True
            return is_jump, len(bkps)
        except Exception as e:
            logger.warning(f"ruptures 运行失败，回退窗口法: {e}")

    logger.info("跳变检测：使用前后窗口中位数差回退方案")
    mad = robust_mad(r)
    thr = JUMP_K * 1.4826 * mad
    n = len(r)
    bkp_count = 0
    for i in range(JUMP_WINDOW, n - JUMP_WINDOW):
        left = r.iloc[i - JUMP_WINDOW : i].median()
        right = r.iloc[i : i + JUMP_WINDOW].median()
        if pd.notna(left) and pd.notna(right) and abs(right - left) > thr:
            is_jump.iloc[i - 1 : i + 2] = True
            bkp_count += 1
    return is_jump, bkp_count


def detect_stuck(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """stuck检测：40分钟窗口联合方差/最大差分/唯一值条件。"""
    r = df["raw_value"]
    n = len(r)
    mask = pd.Series(False, index=df.index)

    for i in range(0, n - STUCK_WINDOW + 1):
        w = r.iloc[i : i + STUCK_WINDOW]
        if w.isna().mean() > 0.2:
            continue
        wv = w.dropna().to_numpy()
        if len(wv) < STUCK_WINDOW - 1:
            continue
        var_ok = np.var(wv) <= STUCK_VAR_THR
        diff_ok = np.max(np.abs(np.diff(wv))) <= STUCK_MAX_DIFF_THR if len(wv) >= 2 else False
        uniq_ok = len(np.unique(np.round(wv, 6))) <= STUCK_UNIQUE_MAX
        # 为避免误判真实平稳段，要求至少满足3个条件中的2个且必须“极平坦”
        if var_ok and ((diff_ok and uniq_ok) or (diff_ok and np.var(wv) < STUCK_VAR_THR * 0.1) or (uniq_ok and np.var(wv) < STUCK_VAR_THR * 0.1)):
            mask.iloc[i : i + STUCK_WINDOW] = True

    segs = contiguous_segments(mask)
    return mask, len(segs)


def conservative_clean(df: pd.DataFrame) -> pd.Series:
    """保守清洗：只修短缺失和孤立尖峰，长异常段保留 NaN。"""
    cleaned = df["raw_value"].copy()

    obs_anomaly = df["is_observation_anomaly"]
    harsh = df["is_stuck"] | df["is_jump"]

    # 长段可疑直接置空，不强修
    cleaned.loc[harsh] = np.nan

    # 孤立尖峰：仅当前后都正常时才做局部插值
    spikes = df["is_spike"].copy()
    for i in np.where(spikes.to_numpy())[0]:
        if 1 <= i < len(cleaned) - 1:
            if not obs_anomaly.iloc[i - 1] and not obs_anomaly.iloc[i + 1] and not harsh.iloc[i]:
                cleaned.iloc[i] = np.nan

    # 短缺失补值（<=SHORT_GAP_MAX）
    miss = cleaned.isna()
    for s, e in contiguous_segments(miss):
        if (e - s + 1) <= SHORT_GAP_MAX:
            cleaned.iloc[s : e + 1] = np.nan

    cleaned = cleaned.interpolate(limit=SHORT_GAP_MAX, limit_direction="both")
    return cleaned


def make_plots(df: pd.DataFrame, file_stem: str, fig_path: Path) -> None:
    """输出诊断图：原始vs清洗、基线漂移、残差与异常标记。"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df["timestamp"], df["raw_value"], label="raw", lw=1)
    axes[0].plot(df["timestamp"], df["cleaned_value"], label="cleaned", lw=1)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{file_stem} 原始与清洗后")

    axes[1].plot(df["timestamp"], df["baseline"], label="baseline", lw=1)
    axes[1].plot(df["timestamp"], df["drift"], label="drift", lw=1)
    axes[1].legend(loc="upper right")
    axes[1].set_title("日基线与漂移")

    axes[2].plot(df["timestamp"], df["residual"], label="residual", lw=1)
    axes[2].legend(loc="upper right")
    axes[2].set_title("残差序列")

    axes[3].plot(df["timestamp"], df["residual"], color="gray", lw=0.8, label="residual")
    axes[3].scatter(df.loc[df["is_spike"], "timestamp"], df.loc[df["is_spike"], "residual"], s=10, label="spike")
    axes[3].scatter(df.loc[df["is_jump"], "timestamp"], df.loc[df["is_jump"], "residual"], s=10, label="jump")
    axes[3].scatter(df.loc[df["is_stuck"], "timestamp"], df.loc[df["is_stuck"], "residual"], s=10, label="stuck")
    axes[3].legend(loc="upper right")
    axes[3].set_title("异常标记位置")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def save_outputs(df: pd.DataFrame, out_csv: Path) -> None:
    """保存单文件结果表。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_root = base_dir / "输出" / "第一题流量处理结果"
    per_file_dir = out_root / "per_file_csv"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    per_file_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "run.log")
    logger.info("开始处理流量数据批任务")
    logger.info(f"LOWESS可用={HAS_LOWESS}, ruptures可用={HAS_RUPTURES}")

    files = find_input_files(base_dir)
    if not files:
        logger.error("未找到xlsx文件，请检查目录：数据/第一题、流量数据/")
        print("共处理0个文件，成功0个，失败0个")
        return

    summary_rows = []
    ok, fail = 0, 0

    for fp in files:
        try:
            raw_df, meta = load_and_prepare_file(fp, logger)
            status_cols = meta["status_cols"]

            aligned = align_to_5min_grid(raw_df, status_cols=status_cols)
            aligned["source_file"] = fp.name

            aligned["is_status_issue"] = _status_issue_mask(aligned, status_cols)
            aligned["is_observation_anomaly"] = (
                aligned["is_missing"]
                | aligned["is_duplicate_timestamp"]
                | aligned["is_time_issue"]
                | aligned["is_status_issue"]
            )

            valid0 = (~aligned["is_observation_anomaly"]) & aligned["raw_value"].notna()
            _, baseline0 = build_daily_baseline(aligned, valid0)
            aligned["baseline"] = baseline0
            aligned["drift"] = estimate_drift(aligned, aligned["baseline"], logger)

            valid1 = (~aligned["is_observation_anomaly"]) & aligned["raw_value"].notna() & aligned["drift"].notna()
            _, baseline1 = build_daily_baseline(aligned.assign(raw_value=aligned["raw_value"] - aligned["drift"]), valid1)
            aligned["baseline"] = baseline1

            aligned["residual"] = compute_residual(aligned)
            aligned["is_spike"] = detect_spikes(aligned)
            aligned["is_jump"], jump_segments = detect_jumps(aligned, logger)
            aligned["is_stuck"], stuck_segments = detect_stuck(aligned)

            aligned["is_candidate_physical_event"] = aligned["is_jump"] & (~aligned["is_status_issue"]) & (~aligned["is_time_issue"])
            aligned["cleaned_value"] = conservative_clean(aligned)

            # 输出字段（题目指定）
            out_cols = [
                "source_file", "timestamp", "raw_value", "baseline", "drift", "residual", "is_missing",
                "is_duplicate_timestamp", "is_time_issue", "is_status_issue", "is_spike", "is_jump",
                "is_stuck", "is_observation_anomaly", "is_candidate_physical_event", "cleaned_value"
            ]
            out_df = aligned[out_cols].copy()

            out_csv = per_file_dir / f"{fp.stem}_result.csv"
            save_outputs(out_df, out_csv)
            make_plots(aligned, fp.stem, fig_dir / f"{fp.stem}_overview.png")

            summary_rows.append(
                {
                    "文件名": fp.name,
                    "原始记录数": meta["raw_rows"],
                    "时间对齐后记录数": len(aligned),
                    "缺失点数": int(aligned["is_missing"].sum()),
                    "重复点数": int(aligned["is_duplicate_timestamp"].sum()),
                    "状态异常点数": int(aligned["is_status_issue"].sum()),
                    "尖峰点数": int(aligned["is_spike"].sum()),
                    "跳变段数": int(jump_segments),
                    "stuck段数": int(stuck_segments),
                    "最终可用点数": int(aligned["cleaned_value"].notna().sum()),
                    "自动识别到的时间列名": meta["time_col"],
                    "自动识别到的主流量列名": meta["flow_col"],
                    "读取工作表": meta["sheet_name"],
                }
            )
            ok += 1
            logger.info(f"完成：{fp.name}")
        except Exception as e:
            fail += 1
            logger.exception(f"失败：{fp.name}, error={e}")

    pd.DataFrame(summary_rows).to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")
    logger.info(f"任务结束：共处理{len(files)}个文件，成功{ok}个，失败{fail}个")
    print(f"共处理{len(files)}个文件，成功{ok}个，失败{fail}个")


if __name__ == "__main__":
    main()
