"""
金融科技創新 - Step 1: 資料讀取、因子計算、IC 計算
================================================
資料來源: taiwan_stock_data.xlsx
  Sheet1: 收盤價(元)      → Momentum 因子
  Sheet2: 市值(百萬元)    → Size 因子
  Sheet3: 股價淨值比-TEJ  → BM 因子

執行後產出: ic_data.csv（三條 IC 時間序列）
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

DATA_FILE = 'taiwan_stock_data.xlsx'

# ============================================================
# 1. 讀取資料
# ============================================================

def load_raw_data(filepath=DATA_FILE):
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names  # ['收盤價(元)', '市值(百萬元)', '股價淨值比-TEJ']
    print(f'工作表: {sheets}')

    def read_sheet(name):
        df = pd.read_excel(filepath, sheet_name=name, header=1, index_col=[0, 1])
        df.index.names = ['代號', '名稱']
        df = df.replace('', np.nan).replace('缺值', np.nan)
        df = df.dropna(how='all')
        df = df.astype(float)
        return df

    price_df = read_sheet(sheets[0])   # 收盤價(元)
    size_df  = read_sheet(sheets[1])   # 市值(百萬元)
    pb_df    = read_sheet(sheets[2])   # 股價淨值比-TEJ

    print(f'股票數: {len(price_df)}，時間欄: {price_df.shape[1]} 期')
    print(f'時間範圍: {list(price_df.columns)[0]} ~ {list(price_df.columns)[-1]}')
    return price_df, size_df, pb_df

# ============================================================
# 2. 因子計算
# ============================================================

def calc_factors(price_df, size_df, pb_df):
    """
    Size    = log(市值 × 1,000,000)
    BM      = 1 / 股價淨值比
    Momentum= ln(收盤價[t-1] / 收盤價[t-12])
    """
    cols = price_df.columns

    # Size：log(市值×1e6)，市值為 0 時 log 產生 -inf，需替換為 NaN
    size_factor = np.log(size_df[cols] * 1_000_000)
    size_factor = size_factor.replace(-np.inf, np.nan).replace(np.inf, np.nan)

    # BM：1/PBR，PBR=0 時產生 inf，需替換為 NaN
    bm_factor = 1.0 / pb_df[cols]
    bm_factor = bm_factor.replace(-np.inf, np.nan).replace(np.inf, np.nan)

    # Momentum: 需要 t-1 和 t-12 的收盤價
    col_list = list(cols)
    mom_factor = pd.DataFrame(np.nan, index=price_df.index, columns=cols)
    for i in range(12, len(col_list)):
        p_t1  = price_df[col_list[i - 1]]   # t-1
        p_t12 = price_df[col_list[i - 12]]  # t-12
        ratio = p_t1 / p_t12
        ratio[ratio <= 0] = np.nan
        mom_factor[col_list[i]] = np.log(ratio)

    return bm_factor, size_factor, mom_factor

# ============================================================
# 3. 月報酬計算（簡單報酬率）
# ============================================================

def calc_returns(price_df):
    """月報酬 = (P[t] - P[t-1]) / P[t-1]"""
    col_list = list(price_df.columns)
    ret = pd.DataFrame(np.nan, index=price_df.index, columns=price_df.columns)
    for i in range(1, len(col_list)):
        p_cur  = price_df[col_list[i]]
        p_prev = price_df[col_list[i - 1]]
        ret[col_list[i]] = (p_cur - p_prev) / p_prev
    return ret

# ============================================================
# 4. 橫截面中位數補值
# ============================================================

def fill_cross_section_median(factor_df, ret_df):
    """
    對每一期（欄），若因子或報酬缺值，
    先找同期兩者都有值的股票計算中位數，再填補
    """
    f_filled = factor_df.copy()
    r_filled = ret_df.copy()

    for col in factor_df.columns:
        f = factor_df[col].copy()
        r = ret_df[col].copy()

        valid = f.notna() & r.notna()
        if valid.sum() < 5:
            continue

        f_med = f[valid].median()
        r_med = r[valid].median()

        f_filled.loc[f.isna(), col] = f_med
        r_filled.loc[r.isna(), col] = r_med

    return f_filled, r_filled

# ============================================================
# 5. 單期 IC 計算（spearman 相關係數）
# ============================================================

def calc_IC_series(factor_df, ret_df, min_stocks=30):
    """
    對每個時間點 t，計算：
        IC[t] = CORREL(factor[t], return[t+1])
    回傳 IC 時間序列（長度 = 欄數 - 1）
    """
    col_list = list(factor_df.columns)
    ic_index = col_list[:-1]  # 最後一期沒有 t+1 報酬
    ic_vals  = []

    for i, col in enumerate(col_list[:-1]):
        next_col = col_list[i + 1]
        f = factor_df[col]
        r = ret_df[next_col]

        valid = f.notna() & r.notna()
        if valid.sum() < min_stocks:
            ic_vals.append(np.nan)
            continue

        ic = f[valid].corr(r[valid], method='spearman')
        ic_vals.append(ic)

    return pd.Series(ic_vals, index=ic_index, name='IC')

# ============================================================
# 6. 主流程：計算並儲存三因子 IC
# ============================================================

if __name__ == '__main__':
    print('=== 載入資料 ===')
    price_df, size_df, pb_df = load_raw_data()

    print('\n=== 計算因子 ===')
    bm_factor, size_factor, mom_factor = calc_factors(price_df, size_df, pb_df)

    print('=== 計算月報酬 ===')
    ret_df = calc_returns(price_df)

    print('=== 補值（中位數）===')
    bm_f,   bm_r   = fill_cross_section_median(bm_factor,   ret_df)
    size_f, size_r = fill_cross_section_median(size_factor, ret_df)
    mom_f,  mom_r  = fill_cross_section_median(mom_factor,  ret_df)

    print('=== 計算 IC 時間序列 ===')
    ic_bm   = calc_IC_series(bm_f,   bm_r)
    ic_size = calc_IC_series(size_f, size_r)
    ic_mom  = calc_IC_series(mom_f,  mom_r)

    # 合併成一個 DataFrame
    ic_df = pd.DataFrame({
        'IC_BM'  : ic_bm,
        'IC_Size': ic_size,
        'IC_Mom' : ic_mom,
    })

    ic_df.to_csv('ic_data.csv')
    print(f'\n已儲存 ic_data.csv，共 {len(ic_df)} 期')
    print(ic_df.tail(10))
    print('\n各因子 IC 統計:')
    print(ic_df.describe().round(4))
