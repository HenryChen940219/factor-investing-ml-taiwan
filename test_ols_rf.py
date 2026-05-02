# -*- coding: utf-8 -*-
"""
快速測試：只跑 OLS + RF，驗證完整流程正確後再跑所有模型
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys; sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

DATA_FILE   = '12173212金融.xlsx'
IC_FILE     = 'ic_data.csv'
N_LAG       = 321
N_START     = 179
N_END       = 312
N_FACTORS   = 3
FACTORS     = ['IC_BM', 'IC_Size', 'IC_Mom']
GROUP_SIZES = [96, 48, 19, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# ── 載入 IC ──────────────────────────────────────────────────
ic_df  = pd.read_csv(IC_FILE, index_col=0)
ic_df  = ic_df[FACTORS].ffill().fillna(0)
ic_arr = ic_df.values
ic_periods = list(ic_df.index)
print(f'IC 資料: {len(ic_df)} 期')

# ── 載入因子 & 報酬 ──────────────────────────────────────────
xl = pd.ExcelFile(DATA_FILE)
sheets = xl.sheet_names

def read_sheet(name):
    df = pd.read_excel(DATA_FILE, sheet_name=name, header=1, index_col=[0, 1])
    df.index.names = ['代號', '名稱']
    return df.replace('', np.nan).replace('缺值', np.nan).astype(float).dropna(how='all')

price_df = read_sheet(sheets[0])
size_df  = read_sheet(sheets[1])
pb_df    = read_sheet(sheets[2])
cols     = price_df.columns
col_list = list(cols)

size_f = np.log(size_df[cols] * 1e6).replace(-np.inf, np.nan).replace(np.inf, np.nan)
bm_f   = (1.0 / pb_df[cols]).replace(-np.inf, np.nan).replace(np.inf, np.nan)
mom_f  = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
for i in range(12, len(col_list)):
    r = price_df[col_list[i-1]] / price_df[col_list[i-12]]
    r[r <= 0] = np.nan
    mom_f[col_list[i]] = np.log(r)
ret = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
for i in range(1, len(col_list)):
    ret[col_list[i]] = (price_df[col_list[i]] - price_df[col_list[i-1]]) / price_df[col_list[i-1]]

factors_data = [bm_f, size_f, mom_f]
print(f'因子/報酬資料: {len(bm_f)} 股票 × {len(col_list)} 期')

assert col_list[N_START] == ic_periods[N_START], '時間不對齊！'
assert col_list[N_END]   == ic_periods[N_END],   '時間不對齊！'
print(f'預測區間: {ic_periods[N_START]} ~ {ic_periods[N_END]}（共 {N_END-N_START+1} 期）')

# ── 特徵函式 ─────────────────────────────────────────────────
def make_feat(ic_arr, n, lag=N_LAG):
    start = max(0, n - lag)
    w = ic_arr[start:n]
    if len(w) < lag:
        w = np.vstack([np.zeros((lag - len(w), N_FACTORS)), w])
    return w.flatten()

# ── 組合建構 ─────────────────────────────────────────────────
def portfolio_ret(factor_vals, ret_vals, pred_ic, g):
    valid = factor_vals.notna() & ret_vals.notna()
    if valid.sum() < g * 2 + 5:
        return np.nan
    f = factor_vals[valid]
    r = ret_vals[valid]
    sorted_f = f.sort_values()
    top_stocks    = sorted_f.index[-g:]
    bottom_stocks = sorted_f.index[:g]
    ret_high = r.loc[top_stocks].mean()
    ret_low  = r.loc[bottom_stocks].mean()
    return (ret_high - ret_low) if pred_ic >= 0 else (ret_low - ret_high)

# ── 績效計算 ─────────────────────────────────────────────────
def calc_perf(rets):
    r = np.array([x for x in rets if not np.isnan(x)])
    if len(r) == 0:
        return dict(win_rate=np.nan, sharpe=np.nan, cum_ret=np.nan)
    return dict(
        win_rate = (r > 0).mean(),
        sharpe   = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan,
        cum_ret  = float(np.prod(1 + r) - 1),
    )

def calc_r2(actuals, preds):
    r2 = {}
    for fi, fname in enumerate(FACTORS):
        yt = np.array(actuals)[:, fi]
        yp = np.array(preds)[:, fi]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2[fname] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2

# ── 跑模型 ───────────────────────────────────────────────────
def run(model_name):
    print(f'\n=== {model_name} ===')
    port  = {g: [] for g in GROUP_SIZES}
    preds = []; actuals = []
    n_total = N_END - N_START + 1

    for idx, n in enumerate(range(N_START, N_END + 1)):
        if idx % 30 == 0:
            print(f'  進度 {idx}/{n_total}', end='\r', flush=True)

        # 建訓練集
        X_tr = [make_feat(ic_arr, i) for i in range(N_LAG, n)]
        y_tr = ic_arr[N_LAG:n]
        x_pred = make_feat(ic_arr, n).reshape(1, -1)

        if len(X_tr) < 10:
            pred = np.zeros(N_FACTORS)
        elif model_name == 'OLS':
            pred = LinearRegression().fit(X_tr, y_tr).predict(x_pred)[0]
        elif model_name == 'RF':
            pred = RandomForestRegressor(100, random_state=42, n_jobs=-1).fit(X_tr, y_tr).predict(x_pred)[0]

        preds.append(pred)
        actuals.append(ic_arr[n])

        sel       = int(np.abs(pred).argmax())
        pred_ic   = pred[sel]
        fac_vals  = factors_data[sel][col_list[n]]
        ret_vals  = ret[col_list[n + 1]]

        for g in GROUP_SIZES:
            port[g].append(portfolio_ret(fac_vals, ret_vals, pred_ic, g))

    print(f'  完成！                    ')

    # 績效彙整
    rows = []
    for g in GROUP_SIZES:
        p = calc_perf(port[g])
        rows.append({'model': model_name, 'g': g, **p})
    df = pd.DataFrame(rows)

    r2 = calc_r2(actuals, preds)
    print(f'  OOS R²: { {k: round(v,4) for k,v in r2.items()} }')
    print(df[['g', 'win_rate', 'sharpe', 'cum_ret']].to_string(index=False))
    return df, preds, actuals

results = []
for m in ['OLS', 'RF']:
    df, preds, actuals = run(m)
    results.append(df)

final = pd.concat(results, ignore_index=True)
final.to_csv('results_ols_rf.csv', index=False, encoding='utf-8-sig')
print('\n已儲存 results_ols_rf.csv')
