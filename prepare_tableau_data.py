# -*- coding: utf-8 -*-
"""
產生 Tableau Public 匯入用的長格式 CSV
=====================================
讀取:
  ic_data.csv          (step1 產出)
  results_portfolio.csv (main_analysis 產出)
  results_r2.csv        (main_analysis 產出)

輸出（皆為 Tableau-friendly 長格式）:
  tableau_ic_series.csv      - IC 時間序列（date × factor）
  tableau_performance.csv    - 模型績效（model × group_size × metric）
  tableau_r2.csv             - 樣本外 R²（model × factor）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# ============================================================
# 1. IC 時間序列 → 長格式
# ============================================================
ic_df = pd.read_csv('ic_data.csv', index_col=0)
ic_df.index.name = 'period'
ic_df = ic_df.reset_index()

# 把 yyyy/mm 轉成 yyyy-mm-01 真正的日期
ic_df['date'] = pd.to_datetime(ic_df['period'].str.replace('/', '-') + '-01',
                                format='%Y-%m-%d', errors='coerce')

ic_long = ic_df.melt(
    id_vars=['date', 'period'],
    value_vars=['IC_BM', 'IC_Size', 'IC_Mom'],
    var_name='factor',
    value_name='ic_value'
)
ic_long['factor'] = ic_long['factor'].str.replace('IC_', '')
ic_long = ic_long.dropna(subset=['ic_value'])
ic_long.to_csv('tableau_ic_series.csv', index=False, encoding='utf-8-sig')
print(f'tableau_ic_series.csv: {len(ic_long)} rows')

# ============================================================
# 2. 模型績效 → 長格式（含每組對應分組數）
# ============================================================
perf = pd.read_csv('results_portfolio.csv')

# 對應 g（每組股票數）→ n_groups（總分組數）的對照（基於約 964 檔股票）
g_to_groups = {1: 964, 2: 482, 3: 321, 4: 241, 5: 192, 6: 160, 7: 137,
               8: 120, 9: 107, 10: 96, 19: 50, 48: 20, 96: 10}
perf['n_groups'] = perf['g'].map(g_to_groups)

perf_long = perf.melt(
    id_vars=['model', 'g', 'n_groups'],
    value_vars=['win_rate', 'sharpe', 'cum_ret'],
    var_name='metric',
    value_name='value'
)
perf_long.columns = ['model', 'group_size', 'n_groups', 'metric', 'value']
perf_long.to_csv('tableau_performance.csv', index=False, encoding='utf-8-sig')
print(f'tableau_performance.csv: {len(perf_long)} rows')

# ============================================================
# 3. 樣本外 R² → 長格式
# ============================================================
r2 = pd.read_csv('results_r2.csv', index_col=0)
r2.index.name = 'model'
r2 = r2.reset_index()

r2_long = r2.melt(
    id_vars=['model'],
    value_vars=['IC_BM', 'IC_Size', 'IC_Mom'],
    var_name='factor',
    value_name='oos_r2'
)
r2_long['factor'] = r2_long['factor'].str.replace('IC_', '')
r2_long.to_csv('tableau_r2.csv', index=False, encoding='utf-8-sig')
print(f'tableau_r2.csv: {len(r2_long)} rows')

print('\n全部輸出完成！可直接匯入 Tableau Public。')
