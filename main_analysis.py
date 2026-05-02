# -*- coding: utf-8 -*-
"""
金融科技創新 - 完整分析主程式
================================
需要先執行: python step1_data_prep.py（產出 ic_data.csv）

本程式跑: OLS / RF / NN1~NN5 / XGBoost / LSTM
輸出:
  results_portfolio.csv  - 各模型×各組數的勝率/夏普/累積報酬
  results_r2.csv         - 各模型的樣本外 R²
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM as KerasLSTM, Input
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping

# ================================================================
# 設定
# ================================================================
DATA_FILE   = '12173212金融.xlsx'
IC_FILE     = 'ic_data.csv'
N_LAG       = 321       # 特徵視窗長度（月），321×3=963≈Input(964)
N_VAL       = 48        # NN 驗證集長度（月）
N_START     = 179       # 第一個預測 IC 索引 → 對應 2013/12
N_END       = 312       # 最後預測 IC 索引  → 對應 2025/01
FACTORS     = ['IC_BM', 'IC_Size', 'IC_Mom']
N_FACTORS   = 3
GROUP_SIZES = [96, 48, 19, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# NN 架構設定（逐漸複雜）
NN_CONFIGS = {
    'NN1': {'layers': [8]},
    'NN2': {'layers': [16, 8]},
    'NN3': {'layers': [16, 8, 4]},
    'NN4': {'layers': [32, 16, 8, 4]},
    'NN5': {'layers': [32, 16, 8, 4, 2], 'use_l1': True},
}

# ================================================================
# 1. 載入資料
# ================================================================
def load_all_data():
    # IC 時間序列
    ic_df = pd.read_csv(IC_FILE, index_col=0)
    ic_df = ic_df[FACTORS].ffill().fillna(0)
    ic_arr = ic_df.values          # shape: (314, 3)
    ic_periods = list(ic_df.index)

    # 原始因子 & 報酬
    xl = pd.ExcelFile(DATA_FILE)
    sheets = xl.sheet_names

    def read(name):
        df = pd.read_excel(DATA_FILE, sheet_name=name, header=1, index_col=[0, 1])
        df.index.names = ['代號', '名稱']
        return df.replace('', np.nan).replace('缺值', np.nan).astype(float).dropna(how='all')

    price_df = read(sheets[0])
    size_df  = read(sheets[1])
    pb_df    = read(sheets[2])
    cols     = price_df.columns
    col_list = list(cols)

    size_f = np.log(size_df[cols] * 1e6).replace(-np.inf, np.nan).replace(np.inf, np.nan)
    bm_f   = (1.0 / pb_df[cols]).replace(-np.inf, np.nan).replace(np.inf, np.nan)

    mom_f = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
    for i in range(12, len(col_list)):
        r = price_df[col_list[i-1]] / price_df[col_list[i-12]]
        r[r <= 0] = np.nan
        mom_f[col_list[i]] = np.log(r)

    ret = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
    for i in range(1, len(col_list)):
        ret[col_list[i]] = (price_df[col_list[i]] - price_df[col_list[i-1]]) / price_df[col_list[i-1]]

    assert col_list[N_START] == ic_periods[N_START]
    assert col_list[N_END]   == ic_periods[N_END]
    print(f'IC: {len(ic_df)} 期 | 因子: {len(bm_f)} 股票 × {len(col_list)} 期')
    print(f'預測區間: {ic_periods[N_START]} ~ {ic_periods[N_END]}（共 {N_END-N_START+1} 期）')
    return ic_arr, ic_periods, [bm_f, size_f, mom_f], ret, col_list

# ================================================================
# 2. 特徵建構
# ================================================================
def make_feat(ic_arr, n, lag=N_LAG):
    """取 n 期之前 lag 期的 IC 值作特徵，不足則補 0"""
    start = max(0, n - lag)
    w = ic_arr[start:n]
    if len(w) < lag:
        w = np.vstack([np.zeros((lag - len(w), N_FACTORS)), w])
    return w.flatten()   # shape: (lag × N_FACTORS,)

def make_seq(ic_arr, n, lag=N_LAG):
    """LSTM 用：回傳 (lag, N_FACTORS) 的序列"""
    start = max(0, n - lag)
    seq = ic_arr[start:n]
    if len(seq) < lag:
        seq = np.vstack([np.zeros((lag - len(seq), N_FACTORS)), seq])
    return seq

# ================================================================
# 3. 模型建構
# ================================================================
def build_nn(n_input, layers, use_l1=False):
    model = Sequential()
    model.add(Input(shape=(n_input,)))
    for i, units in enumerate(layers):
        reg = l1(1e-4) if (use_l1 and i == 0) else None
        model.add(Dense(units, activation='relu', kernel_regularizer=reg))
    model.add(Dense(N_FACTORS))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm():
    model = Sequential([
        Input(shape=(N_LAG, N_FACTORS)),
        KerasLSTM(32, activation='tanh'),
        Dense(N_FACTORS)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ================================================================
# 4. 單期預測函式
# ================================================================
def pred_ols(ic_arr, n):
    # 訓練集從 period 1 開始（make_feat 已處理不足 N_LAG 的補零）
    X = [make_feat(ic_arr, i) for i in range(1, n)]
    y = ic_arr[1:n]
    if len(X) < 10: return np.zeros(N_FACTORS)
    return LinearRegression().fit(X, y).predict(make_feat(ic_arr, n).reshape(1, -1))[0]

def pred_rf(ic_arr, n):
    X = [make_feat(ic_arr, i) for i in range(1, n)]
    y = ic_arr[1:n]
    if len(X) < 10: return np.zeros(N_FACTORS)
    m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return m.fit(X, y).predict(make_feat(ic_arr, n).reshape(1, -1))[0]

def pred_xgb(ic_arr, n):
    X = np.array([make_feat(ic_arr, i) for i in range(1, n)])
    y = ic_arr[1:n]
    if len(X) < 10: return np.zeros(N_FACTORS)
    preds = []
    for f in range(N_FACTORS):
        m = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        m.fit(X, y[:, f])
        preds.append(m.predict(make_feat(ic_arr, n).reshape(1, -1))[0])
    return np.array(preds)

def pred_nn(ic_arr, n, nn_name):
    cfg     = NN_CONFIGS[nn_name]
    n_input = N_LAG * N_FACTORS
    X_all   = np.array([make_feat(ic_arr, i) for i in range(1, n)])
    y_all   = ic_arr[1:n]
    if len(X_all) < 10: return np.zeros(N_FACTORS)

    n_split  = max(5, len(X_all) - N_VAL)
    X_tr, y_tr = X_all[:n_split], y_all[:n_split]
    X_vl, y_vl = X_all[n_split:], y_all[n_split:]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_vl_s = sc.transform(X_vl) if len(X_vl) > 0 else X_vl
    x_p    = sc.transform(make_feat(ic_arr, n).reshape(1, -1))

    model = build_nn(n_input, cfg['layers'], cfg.get('use_l1', False))
    cb    = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    val_data = (X_vl_s, y_vl) if len(X_vl) > 0 else None
    model.fit(X_tr_s, y_tr, validation_data=val_data,
              epochs=100, batch_size=16, verbose=0, callbacks=[cb])
    return model.predict(x_p, verbose=0)[0]

def pred_lstm(ic_arr, n):
    X_all = np.array([make_seq(ic_arr, i) for i in range(1, n)])
    y_all = ic_arr[1:n]
    if len(X_all) < 10: return np.zeros(N_FACTORS)

    n_split  = max(5, len(X_all) - N_VAL)
    X_tr, y_tr = X_all[:n_split], y_all[:n_split]
    X_vl, y_vl = X_all[n_split:], y_all[n_split:]

    model = build_lstm()
    cb    = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    val_data = (X_vl, y_vl) if len(X_vl) > 0 else None
    model.fit(X_tr, y_tr, validation_data=val_data,
              epochs=100, batch_size=16, verbose=0, callbacks=[cb])
    return model.predict(make_seq(ic_arr, n).reshape(1, N_LAG, N_FACTORS), verbose=0)[0]

# ================================================================
# 5. 投資組合 & 績效
# ================================================================
def portfolio_ret(factor_vals, ret_vals, pred_ic, g):
    valid = factor_vals.notna() & ret_vals.notna()
    if valid.sum() < g * 2 + 5: return np.nan
    f = factor_vals[valid]
    r = ret_vals[valid]
    sf = f.sort_values()
    ret_high = r.loc[sf.index[-g:]].mean()
    ret_low  = r.loc[sf.index[:g]].mean()
    return (ret_high - ret_low) if pred_ic >= 0 else (ret_low - ret_high)

def calc_perf(rets):
    r = np.array([x for x in rets if not np.isnan(x)])
    if len(r) == 0: return dict(win_rate=np.nan, sharpe=np.nan, cum_ret=np.nan)
    return dict(
        win_rate = float((r > 0).mean()),
        sharpe   = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan,
        cum_ret  = float(np.prod(1 + r) - 1),
    )

def calc_r2(actuals, preds):
    r2 = {}
    for fi, fname in enumerate(FACTORS):
        yt = np.array(actuals)[:, fi]
        yp = np.array(preds)[:, fi]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2[fname] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return r2

# ================================================================
# 6. 主迴圈
# ================================================================
def run_model(model_name, ic_arr, factors_data, ret, col_list):
    print(f'\n=== [{model_name}] 開始 ===')
    port    = {g: [] for g in GROUP_SIZES}
    preds   = []
    actuals = []
    n_total = N_END - N_START + 1

    for idx, n in enumerate(range(N_START, N_END + 1)):
        if idx % 20 == 0:
            print(f'  進度 {idx}/{n_total}', flush=True)

        # 預測 IC
        if   model_name == 'OLS':    pred = pred_ols(ic_arr, n)
        elif model_name == 'RF':     pred = pred_rf(ic_arr, n)
        elif model_name == 'XGBoost':pred = pred_xgb(ic_arr, n)
        elif model_name in NN_CONFIGS: pred = pred_nn(ic_arr, n, model_name)
        elif model_name == 'LSTM':   pred = pred_lstm(ic_arr, n)
        else: raise ValueError(model_name)

        preds.append(pred)
        actuals.append(ic_arr[n])

        # 選因子、建組合
        sel       = int(np.abs(pred).argmax())
        pred_ic   = pred[sel]
        fac_vals  = factors_data[sel][col_list[n]]
        ret_vals  = ret[col_list[n + 1]]
        for g in GROUP_SIZES:
            port[g].append(portfolio_ret(fac_vals, ret_vals, pred_ic, g))

    print(f'  [{model_name}] 完成！')

    # 整理輸出
    rows = []
    for g in GROUP_SIZES:
        p = calc_perf(port[g])
        rows.append({'model': model_name, 'g': g, **p})
    df = pd.DataFrame(rows)

    r2 = calc_r2(actuals, preds)
    print(f'  OOS R²: { {k: round(v, 4) for k, v in r2.items()} }')
    print(df[['g', 'win_rate', 'sharpe', 'cum_ret']].to_string(index=False))
    return df, r2

# ================================================================
# 7. 執行
# ================================================================
if __name__ == '__main__':
    print('========== 載入資料 ==========')
    ic_arr, ic_periods, factors_data, ret, col_list = load_all_data()

    MODEL_LIST = ['OLS', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'XGBoost', 'LSTM']

    all_perf = []
    all_r2   = {}

    for model_name in MODEL_LIST:
        df, r2 = run_model(model_name, ic_arr, factors_data, ret, col_list)
        all_perf.append(df)
        all_r2[model_name] = r2

    # 儲存結果
    final_perf = pd.concat(all_perf, ignore_index=True)
    final_perf.to_csv('results_portfolio.csv', index=False, encoding='utf-8-sig')

    r2_df = pd.DataFrame(all_r2).T
    r2_df.index.name = 'model'
    r2_df.to_csv('results_r2.csv', encoding='utf-8-sig')

    print('\n\n========== 最終績效彙整 ==========')
    print(final_perf.pivot_table(values='sharpe', index='model', columns='g').round(3).to_string())

    print('\n========== 樣本外 R² ==========')
    print(r2_df.round(4).to_string())

    print('\n已儲存 results_portfolio.csv 與 results_r2.csv')
