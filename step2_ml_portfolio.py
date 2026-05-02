"""
金融科技創新 - Step 2: ML 模型預測 IC + 投資組合建構
=====================================================
讀取: ic_data.csv (step1 產出)、12173212金融.xlsx (因子排序用)

流程:
  1. 特徵工程: 以過去 N_LAG 期 IC 值作為輸入特徵
  2. 滾動訓練: 每期用 expanding window 訓練，預測下一期 IC
  3. 選股: 選 |IC_pred| 最大的因子 → 排序股票 → 多空組合
  4. 績效: 勝率、夏普比率、累積報酬、樣本外 R²

模型列表:
  OLS / RF / NN1~NN5 / XGBoost / LSTM
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from scipy import stats

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

# ============================================================
# 設定
# ============================================================

DATA_FILE  = '12173212金融.xlsx'
IC_FILE    = 'ic_data.csv'

N_LAG      = 321   # 用前 N_LAG 期 IC 作特徵（321×3=963≈Input(964)，與報告一致）
N_VAL      = 48    # NN 驗證集長度（最近 48 期）
N_START    = 179   # 第一個預測期的 IC 索引（對應 2013/12）
N_END      = 312   # 最後一個預測期的 IC 索引（對應 2025/01）
MIN_STOCKS = 30    # 每期最少有效股票數
GROUP_SIZES = [96, 48, 19, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

FACTORS    = ['IC_BM', 'IC_Size', 'IC_Mom']
N_FACTORS  = len(FACTORS)

# ============================================================
# 1. 載入 IC 資料
# ============================================================

def load_ic(filepath=IC_FILE):
    df = pd.read_csv(filepath, index_col=0)
    df = df[FACTORS]
    df = df.ffill().fillna(0)   # NaN 先向前填，剩餘補 0
    print(f'IC 資料: {len(df)} 期，範圍 {df.index[0]} ~ {df.index[-1]}')
    return df

# ============================================================
# 2. 載入因子值與報酬（用於選股排序）
# ============================================================

def load_factor_and_ret(filepath=DATA_FILE):
    """回傳 bm_f, size_f, mom_f, ret：DataFrame(股票×時間)"""
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names

    def read_sheet(name):
        df = pd.read_excel(filepath, sheet_name=name, header=1, index_col=[0, 1])
        df.index.names = ['代號', '名稱']
        df = df.replace('', np.nan).replace('缺值', np.nan).astype(float)
        return df.dropna(how='all')

    price_df = read_sheet(sheets[0])
    size_df  = read_sheet(sheets[1])
    pb_df    = read_sheet(sheets[2])
    cols     = price_df.columns
    col_list = list(cols)

    # 因子計算
    size_f = np.log(size_df[cols] * 1_000_000)
    size_f = size_f.replace(-np.inf, np.nan).replace(np.inf, np.nan)

    bm_f = 1.0 / pb_df[cols]
    bm_f = bm_f.replace(-np.inf, np.nan).replace(np.inf, np.nan)

    mom_f = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
    for i in range(12, len(col_list)):
        p1  = price_df[col_list[i - 1]]
        p12 = price_df[col_list[i - 12]]
        ratio = p1 / p12
        ratio[ratio <= 0] = np.nan
        mom_f[col_list[i]] = np.log(ratio)

    # 月報酬
    ret = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
    for i in range(1, len(col_list)):
        ret[col_list[i]] = (price_df[col_list[i]] - price_df[col_list[i-1]]) / price_df[col_list[i-1]]

    print(f'因子資料: {len(bm_f)} 股票 × {len(col_list)} 期')
    return bm_f, size_f, mom_f, ret, col_list

# ============================================================
# 3. 特徵工程
# ============================================================

def make_features(ic_arr, n, lag=N_LAG):
    """
    從 IC 矩陣（shape: T × 3）取第 n 期預測所需的特徵向量
    features = [IC_BM[n-lag..n-1], IC_Size[n-lag..n-1], IC_Mom[n-lag..n-1]]
    長度 = lag × 3
    """
    start = max(0, n - lag)
    window = ic_arr[start:n]                      # shape: (actual_lag, 3)
    # 若歷史不足 lag 期，在開頭補 0
    if len(window) < lag:
        pad = np.zeros((lag - len(window), N_FACTORS))
        window = np.vstack([pad, window])
    return window.flatten()                        # shape: (lag*3,)

def build_dataset(ic_arr, n_start, n_end, lag=N_LAG):
    """建立整段訓練特徵矩陣 (n_end - n_start + 1 筆)"""
    X, y = [], []
    for n in range(n_start, n_end + 1):
        X.append(make_features(ic_arr, n, lag))
        y.append(ic_arr[n])
    return np.array(X), np.array(y)

# ============================================================
# 4. 模型定義
# ============================================================

def build_nn(n_input, n_output, layers, use_l1=False):
    model = Sequential()
    model.add(Input(shape=(n_input,)))
    for i, units in enumerate(layers):
        reg = l1(1e-4) if (use_l1 and i == 0) else None
        model.add(Dense(units, activation='relu', kernel_regularizer=reg))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm(seq_len, n_feat, n_output):
    model = Sequential([
        Input(shape=(seq_len, n_feat)),
        KerasLSTM(32, activation='tanh'),
        Dense(n_output)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# NN 架構（逐漸複雜）
NN_CONFIGS = {
    'NN1': {'layers': [8]},
    'NN2': {'layers': [16, 8]},
    'NN3': {'layers': [16, 8, 4]},
    'NN4': {'layers': [32, 16, 8, 4]},
    'NN5': {'layers': [32, 16, 8, 4, 2], 'use_l1': True},
}

# ============================================================
# 5. 單期預測
# ============================================================

def predict_ic_ols(ic_arr, n, lag=N_LAG):
    X_train = [make_features(ic_arr, i, lag) for i in range(lag, n)]
    y_train = ic_arr[lag:n]
    if len(X_train) < 10:
        return np.zeros(N_FACTORS)
    model = LinearRegression()
    model.fit(X_train, y_train)
    x_pred = make_features(ic_arr, n, lag).reshape(1, -1)
    return model.predict(x_pred)[0]

def predict_ic_rf(ic_arr, n, lag=N_LAG):
    X_train = [make_features(ic_arr, i, lag) for i in range(lag, n)]
    y_train = ic_arr[lag:n]
    if len(X_train) < 10:
        return np.zeros(N_FACTORS)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    x_pred = make_features(ic_arr, n, lag).reshape(1, -1)
    return model.predict(x_pred)[0]

def predict_ic_xgb(ic_arr, n, lag=N_LAG):
    X_train = [make_features(ic_arr, i, lag) for i in range(lag, n)]
    y_train = ic_arr[lag:n]
    if len(X_train) < 10:
        return np.zeros(N_FACTORS)
    preds = []
    for f in range(N_FACTORS):
        model = xgb.XGBRegressor(n_estimators=100, random_state=42,
                                  verbosity=0, use_label_encoder=False)
        model.fit(np.array(X_train), y_train[:, f])
        x_pred = make_features(ic_arr, n, lag).reshape(1, -1)
        preds.append(model.predict(x_pred)[0])
    return np.array(preds)

def predict_ic_nn(ic_arr, n, nn_name, lag=N_LAG, n_val=N_VAL):
    cfg = NN_CONFIGS[nn_name]
    n_input = lag * N_FACTORS

    # 訓練集：除最後 n_val 期外的所有資料
    n_train_end = n - n_val
    if n_train_end <= lag + 10:
        n_train_end = lag + 10   # 保留最少訓練量

    X_all  = [make_features(ic_arr, i, lag) for i in range(lag, n)]
    y_all  = ic_arr[lag:n]

    n_split  = max(0, len(X_all) - n_val)
    X_train  = np.array(X_all[:n_split])
    y_train  = y_all[:n_split]
    X_val    = np.array(X_all[n_split:])
    y_val    = y_all[n_split:]

    if len(X_train) < 5:
        return np.zeros(N_FACTORS)

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_val_s   = sc.transform(X_val)
    x_pred_s  = sc.transform(make_features(ic_arr, n, lag).reshape(1, -1))

    model = build_nn(n_input, N_FACTORS,
                     cfg['layers'], cfg.get('use_l1', False))
    cb = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train_s, y_train,
              validation_data=(X_val_s, y_val) if len(X_val) > 0 else None,
              epochs=100, batch_size=16, verbose=0, callbacks=[cb])
    return model.predict(x_pred_s, verbose=0)[0]

def predict_ic_lstm(ic_arr, n, lag=N_LAG, n_val=N_VAL):
    """LSTM：input shape = (lag, N_FACTORS)"""
    def make_seq(ic_arr, n, lag):
        start = max(0, n - lag)
        seq   = ic_arr[start:n]
        if len(seq) < lag:
            pad = np.zeros((lag - len(seq), N_FACTORS))
            seq = np.vstack([pad, seq])
        return seq  # shape: (lag, 3)

    # 建立訓練樣本
    X_seq, y_seq = [], []
    for i in range(lag, n):
        X_seq.append(make_seq(ic_arr, i, lag))
        y_seq.append(ic_arr[i])

    if len(X_seq) < 10:
        return np.zeros(N_FACTORS)

    X_arr = np.array(X_seq)   # (T, lag, 3)
    y_arr = np.array(y_seq)   # (T, 3)

    n_split = max(0, len(X_arr) - n_val)
    X_tr, y_tr = X_arr[:n_split], y_arr[:n_split]
    X_vl, y_vl = X_arr[n_split:], y_arr[n_split:]

    if len(X_tr) < 5:
        return np.zeros(N_FACTORS)

    model = build_lstm(lag, N_FACTORS, N_FACTORS)
    cb    = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr,
              validation_data=(X_vl, y_vl) if len(X_vl) > 0 else None,
              epochs=100, batch_size=16, verbose=0, callbacks=[cb])
    x_pred = make_seq(ic_arr, n, lag).reshape(1, lag, N_FACTORS)
    return model.predict(x_pred, verbose=0)[0]

# ============================================================
# 6. 投資組合建構（單期）
# ============================================================

def portfolio_return_single(factor_vals, ret_vals, pred_ic, g):
    """
    factor_vals: 所有股票在 t 期的因子值 (Series)
    ret_vals   : 所有股票在 t+1 期的報酬 (Series)
    pred_ic    : 預測 IC（正→買高賣低；負→買低賣高）
    g          : 每組股票數

    回傳: 多空組合報酬
    """
    # 只取兩者都有值的股票
    valid = factor_vals.notna() & ret_vals.notna()
    if valid.sum() < g * 2 + 5:
        return np.nan

    f = factor_vals[valid]
    r = ret_vals[valid]

    sorted_stocks = f.sort_values()
    bottom_stocks = sorted_stocks.index[:g]
    top_stocks    = sorted_stocks.index[-g:]

    ret_high = r.loc[top_stocks].mean()
    ret_low  = r.loc[bottom_stocks].mean()

    if pred_ic >= 0:
        return ret_high - ret_low
    else:
        return ret_low - ret_high

# ============================================================
# 7. 績效評估
# ============================================================

def calc_performance(ret_series):
    """輸入: 報酬時間序列 (array-like)，回傳 dict"""
    r = np.array(ret_series)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {'win_rate': np.nan, 'sharpe': np.nan,
                'cum_ret': np.nan}
    win_rate = (r > 0).mean()
    sharpe   = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    cum_ret  = np.prod(1 + r) - 1
    return {'win_rate': win_rate, 'sharpe': sharpe, 'cum_ret': cum_ret}

def calc_oos_r2(y_true, y_pred):
    """樣本外 R²（每個因子分別計算）"""
    results = {}
    for f in range(N_FACTORS):
        yt = np.array(y_true)[:, f]
        yp = np.array(y_pred)[:, f]
        valid = ~(np.isnan(yt) | np.isnan(yp))
        if valid.sum() < 5:
            results[FACTORS[f]] = np.nan
            continue
        ss_res = np.sum((yt[valid] - yp[valid]) ** 2)
        ss_tot = np.sum((yt[valid] - yt[valid].mean()) ** 2)
        results[FACTORS[f]] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return results

# ============================================================
# 8. 主迴圈：滾動預測 + 建構組合
# ============================================================

def run_model(model_name, ic_arr, factors_data, ret_data, col_list,
              n_start=N_START, n_end=N_END):
    """
    model_name: 'OLS', 'RF', 'NN1'~'NN5', 'XGBoost', 'LSTM'
    回傳:
      portfolio_rets: dict { g: [報酬序列] }
      ic_preds      : list of predicted IC vectors
      ic_actuals    : list of actual IC vectors
    """
    print(f'  [{model_name}] 開始預測...', flush=True)

    portfolio_rets = {g: [] for g in GROUP_SIZES}
    ic_preds       = []
    ic_actuals     = []

    n_total = n_end - n_start + 1

    for idx, n in enumerate(range(n_start, n_end + 1)):
        if idx % 20 == 0:
            print(f'    進度 {idx}/{n_total}', flush=True)

        # (a) 預測 IC
        if model_name == 'OLS':
            pred = predict_ic_ols(ic_arr, n)
        elif model_name == 'RF':
            pred = predict_ic_rf(ic_arr, n)
        elif model_name == 'XGBoost':
            pred = predict_ic_xgb(ic_arr, n)
        elif model_name in NN_CONFIGS:
            pred = predict_ic_nn(ic_arr, n, model_name)
        elif model_name == 'LSTM':
            pred = predict_ic_lstm(ic_arr, n)
        else:
            raise ValueError(f'Unknown model: {model_name}')

        ic_preds.append(pred)
        ic_actuals.append(ic_arr[n])

        # (b) 選擇因子（|pred_IC| 最大者）
        sel_factor = int(np.argmax(np.abs(pred)))  # 0=BM, 1=Size, 2=Mom
        pred_ic_val = pred[sel_factor]

        # (c) 取得因子值與下期報酬
        period_col      = col_list[n]       # 當期欄位（用於排序）
        next_period_col = col_list[n + 1]   # 下期欄位（用於計算報酬）

        fac_map = [factors_data[0], factors_data[1], factors_data[2]]
        factor_vals = fac_map[sel_factor][period_col]
        ret_vals    = ret_data[next_period_col]

        # (d) 計算各組合報酬
        for g in GROUP_SIZES:
            pr = portfolio_return_single(factor_vals, ret_vals, pred_ic_val, g)
            portfolio_rets[g].append(pr)

    print(f'  [{model_name}] 完成！', flush=True)
    return portfolio_rets, ic_preds, ic_actuals

# ============================================================
# 9. 彙整輸出
# ============================================================

def summarize_results(model_name, portfolio_rets, ic_preds, ic_actuals):
    rows = []
    for g in GROUP_SIZES:
        perf = calc_performance(portfolio_rets[g])
        n_groups_approx = round(1000 / g)  # 大約等份數
        rows.append({
            'model'    : model_name,
            'g'        : g,
            'n_groups' : n_groups_approx,
            'win_rate' : perf['win_rate'],
            'sharpe'   : perf['sharpe'],
            'cum_ret'  : perf['cum_ret'],
        })
    perf_df = pd.DataFrame(rows)

    r2 = calc_oos_r2(ic_actuals, ic_preds)
    print(f'\n  [{model_name}] 樣本外 R²: {r2}')
    print(perf_df.to_string(index=False))

    return perf_df, r2

# ============================================================
# 10. 主程式
# ============================================================

if __name__ == '__main__':
    # --- 載入資料 ---
    print('=== 載入 IC 資料 ===')
    ic_df  = load_ic()
    ic_arr = ic_df.values    # shape: (314, 3)

    print('\n=== 載入因子與報酬資料 ===')
    bm_f, size_f, mom_f, ret_df, col_list = load_factor_and_ret()
    factors_data = [bm_f, size_f, mom_f]

    # --- 驗證索引對應 ---
    ic_periods    = list(ic_df.index)   # IC 時間索引
    factor_cols   = col_list            # 因子/報酬時間索引（比 IC 多最後一期）
    assert ic_periods[N_START] in factor_cols, \
        f'N_START 索引 {N_START} 對應期間 {ic_periods[N_START]} 不在因子欄位中！'
    assert ic_periods[N_END] in factor_cols, \
        f'N_END 索引 {N_END} 對應期間 {ic_periods[N_END]} 不在因子欄位中！'
    assert factor_cols.index(ic_periods[N_START]) + 1 < len(factor_cols), \
        'N_END 之後需要至少一期報酬！'

    # 確保 col_list 和 ic 時間對齊（factor period n = col_list[n]）
    for n in [N_START, N_END]:
        assert ic_periods[n] == factor_cols[n], \
            f'IC 索引 {n}: IC={ic_periods[n]}, factor={factor_cols[n]}'

    print(f'\n預測區間: {ic_periods[N_START]} ~ {ic_periods[N_END]} (共 {N_END-N_START+1} 期)')

    # --- 跑各模型 ---
    MODEL_LIST = ['OLS', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'XGBoost', 'LSTM']

    all_perf = []
    all_r2   = {}

    for model_name in MODEL_LIST:
        print(f'\n=== 模型: {model_name} ===')
        port_rets, preds, actuals = run_model(
            model_name, ic_arr, factors_data, ret_df, factor_cols)
        perf_df, r2 = summarize_results(model_name, port_rets, preds, actuals)
        all_perf.append(perf_df)
        all_r2[model_name] = r2

    # --- 彙整輸出 ---
    final_perf = pd.concat(all_perf, ignore_index=True)
    final_perf.to_csv('results_portfolio.csv', index=False, encoding='utf-8-sig')
    print('\n=== 已儲存 results_portfolio.csv ===')

    # R² 彙整
    r2_df = pd.DataFrame(all_r2).T
    r2_df.index.name = 'model'
    r2_df.to_csv('results_r2.csv', encoding='utf-8-sig')
    print('=== 已儲存 results_r2.csv ===')

    print('\n\n======== 最終績效彙整 ========')
    print(final_perf.to_string(index=False))
    print('\n======== 樣本外 R² ========')
    print(r2_df.round(4).to_string())
