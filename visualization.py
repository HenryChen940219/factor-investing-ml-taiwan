# -*- coding: utf-8 -*-
"""
金融科技創新 - 視覺化
需先執行 step1_data_prep.py 與 main_analysis.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# ── 中文字體 ─────────────────────────────────────────────────
plt.rcParams['font.family']      = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi']       = 150

# ── 常數 ─────────────────────────────────────────────────────
DATA_FILE   = '12173212金融.xlsx'
IC_FILE     = 'ic_data.csv'
PERF_FILE   = 'results_portfolio.csv'
R2_FILE     = 'results_r2.csv'
N_START     = 179
N_END       = 312
FACTORS     = ['IC_BM', 'IC_Size', 'IC_Mom']
GROUP_SIZES = [96, 48, 19, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
MODEL_ORDER = ['OLS', 'RF', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'XGBoost', 'LSTM']
COLORS      = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336',
               '#00BCD4','#8BC34A','#FF5722','#607D8B']

# ── 載入資料 ─────────────────────────────────────────────────
ic_df   = pd.read_csv(IC_FILE, index_col=0)
perf_df = pd.read_csv(PERF_FILE)
r2_df   = pd.read_csv(R2_FILE, index_col=0)

# 計算各模型各期的投資組合報酬（用於畫累積報酬線）
def load_factor_ret():
    xl = pd.ExcelFile(DATA_FILE)
    sheets = xl.sheet_names
    def read(name):
        df = pd.read_excel(DATA_FILE, sheet_name=name, header=1, index_col=[0,1])
        df.index.names = ['代號','名稱']
        return df.replace('',np.nan).replace('缺值',np.nan).astype(float).dropna(how='all')
    price_df = read(sheets[0])
    size_df  = read(sheets[1])
    pb_df    = read(sheets[2])
    cols     = list(price_df.columns)
    size_f = np.log(size_df[price_df.columns]*1e6).replace(-np.inf,np.nan).replace(np.inf,np.nan)
    bm_f   = (1.0/pb_df[price_df.columns]).replace(-np.inf,np.nan).replace(np.inf,np.nan)
    mom_f  = pd.DataFrame(np.nan, index=price_df.index, columns=price_df.columns)
    for i in range(12, len(cols)):
        r = price_df[cols[i-1]]/price_df[cols[i-12]]; r[r<=0]=np.nan
        mom_f[cols[i]] = np.log(r)
    ret = pd.DataFrame(np.nan, index=price_df.index, columns=price_df.columns)
    for i in range(1, len(cols)):
        ret[cols[i]] = (price_df[cols[i]]-price_df[cols[i-1]])/price_df[cols[i-1]]
    return [bm_f, size_f, mom_f], ret, cols

# ================================================================
# 圖1：IC 時間序列
# ================================================================
def plot_ic_series():
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    factor_names = ['BM（帳面市值比）', 'Size（規模）', 'Mom（動能）']
    colors_ic = ['#1565C0', '#2E7D32', '#E65100']

    for i, (col, name, color) in enumerate(zip(FACTORS, factor_names, colors_ic)):
        ax = axes[i]
        series = ic_df[col].dropna()
        ax.plot(range(len(series)), series.values, color=color, linewidth=0.8, alpha=0.9)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.fill_between(range(len(series)), series.values, 0,
                        where=series.values > 0, color=color, alpha=0.15)
        ax.fill_between(range(len(series)), series.values, 0,
                        where=series.values < 0, color='gray', alpha=0.1)
        mean_ic = series.mean()
        ax.axhline(mean_ic, color=color, linewidth=1, linestyle=':', alpha=0.7)
        ax.set_ylabel('IC 值', fontsize=10)
        ax.set_title(f'{name}  |  平均 IC = {mean_ic:.4f}', fontsize=11, fontweight='bold')
        ax.set_ylim(-0.5, 0.7)
        ax.grid(axis='y', alpha=0.3)

        # 標記 X 軸年份
        years = ['1999','2003','2007','2011','2015','2019','2023']
        year_idx = []
        for yr in years:
            matches = [j for j, idx in enumerate(series.index) if str(idx).startswith(yr)]
            if matches: year_idx.append((matches[0], yr))
        ax.set_xticks([x[0] for x in year_idx])
        ax.set_xticklabels([x[1] for x in year_idx], fontsize=9)

    # 標記 ML 測試起始點
    test_start_idx = list(ic_df[FACTORS[0]].dropna().index).index(ic_df.index[N_START]) \
        if ic_df.index[N_START] in ic_df[FACTORS[0]].dropna().index else N_START
    for ax in axes:
        ax.axvline(test_start_idx, color='red', linewidth=1.2, linestyle='--', alpha=0.7)
    axes[0].annotate('ML 測試起始\n(2013/12)', xy=(test_start_idx, 0.55),
                     xytext=(test_start_idx+8, 0.6),
                     fontsize=8.5, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1))

    fig.suptitle('三因子 IC 時間序列（1999/01 ~ 2025/02）', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('fig1_IC_series.png', bbox_inches='tight')
    plt.close()
    print('已儲存 fig1_IC_series.png')

# ================================================================
# 圖2：勝率熱力圖
# ================================================================
def plot_heatmap_winrate():
    pivot = perf_df.pivot_table(values='win_rate', index='model', columns='g')
    pivot = pivot.reindex(MODEL_ORDER)[sorted(GROUP_SIZES)]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    norm = TwoSlopeNorm(vmin=0.40, vcenter=0.50, vmax=0.65)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                norm=norm, ax=ax, linewidths=0.4,
                cbar_kws={'label': '勝率', 'shrink': 0.8})
    ax.set_title('各模型 × 各組數 勝率（Win Rate）', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('每組股票數 (g)', fontsize=11)
    ax.set_ylabel('模型', fontsize=11)
    ax.set_xticklabels([str(int(c)) for c in sorted(GROUP_SIZES)], rotation=0)
    plt.tight_layout()
    fig.savefig('fig2_winrate_heatmap.png', bbox_inches='tight')
    plt.close()
    print('已儲存 fig2_winrate_heatmap.png')

# ================================================================
# 圖3：夏普比率熱力圖
# ================================================================
def plot_heatmap_sharpe():
    pivot = perf_df.pivot_table(values='sharpe', index='model', columns='g')
    pivot = pivot.reindex(MODEL_ORDER)[sorted(GROUP_SIZES)]

    vmax = min(pivot.max().max(), 0.8)
    vmin = max(pivot.min().min(), -0.8)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                norm=norm, ax=ax, linewidths=0.4,
                cbar_kws={'label': '夏普比率', 'shrink': 0.8})
    ax.set_title('各模型 × 各組數 夏普比率（Sharpe Ratio，年化）', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('每組股票數 (g)', fontsize=11)
    ax.set_ylabel('模型', fontsize=11)
    ax.set_xticklabels([str(int(c)) for c in sorted(GROUP_SIZES)], rotation=0)
    plt.tight_layout()
    fig.savefig('fig3_sharpe_heatmap.png', bbox_inches='tight')
    plt.close()
    print('已儲存 fig3_sharpe_heatmap.png')

# ================================================================
# 圖4：累積報酬曲線（g=96，代表分散型組合）
# ================================================================
def plot_cumret_curves():
    factors_data, ret_df, col_list = load_factor_ret()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, g_target in enumerate([96, 4]):
        ax = axes[ax_idx]
        subset = perf_df[perf_df['g'] == g_target].set_index('model')

        # 重新算各期報酬序列（從 results_portfolio.csv 沒有逐期資料，改用已知累積報酬標示）
        # 這裡用各模型的累積報酬數值作橫向比較長條圖
        cum_rets = []
        for model in MODEL_ORDER:
            if model in subset.index:
                cum_rets.append(subset.loc[model, 'cum_ret'])
            else:
                cum_rets.append(np.nan)

        colors_bar = ['#4CAF50' if v > 0 else '#F44336' for v in cum_rets]
        bars = ax.bar(MODEL_ORDER, cum_rets, color=colors_bar, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f'累積報酬比較（g = {g_target}，每組 {g_target} 檔股票）',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('累積報酬率', fontsize=10)
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha='right', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, cum_rets):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + (0.02 if val >= 0 else -0.05),
                        f'{val*100:.1f}%', ha='center', va='bottom', fontsize=7.5)

    fig.suptitle('各模型累積報酬率比較（2013/12 ~ 2025/01）', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig('fig4_cumret_bars.png', bbox_inches='tight')
    plt.close()
    print('已儲存 fig4_cumret_bars.png')

# ================================================================
# 圖5：樣本外 R² 長條圖
# ================================================================
def plot_r2():
    r2_plot = r2_df.reindex(MODEL_ORDER).clip(lower=-5)   # 裁掉極端值方便展示
    clipped = r2_df.reindex(MODEL_ORDER).min().min() < -5

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.26
    bar_colors = ['#1565C0', '#2E7D32', '#E65100']
    factor_labels = ['BM', 'Size', 'Mom']

    for i, (col, label, color) in enumerate(zip(FACTORS, factor_labels, bar_colors)):
        vals = r2_plot[col].values
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=f'IC_{label}', color=color, alpha=0.82, edgecolor='white')
        for bar, orig_val in zip(bars, r2_df.reindex(MODEL_ORDER)[col].values):
            if orig_val < -5:
                ax.text(bar.get_x() + bar.get_width()/2, -4.8,
                        f'{orig_val:.0f}', ha='center', va='bottom',
                        fontsize=6.5, color=color, rotation=90)

    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10)
    ax.set_ylabel('樣本外 R²', fontsize=11)
    ax.set_ylim(-5.5, 0.3)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    title = '各模型樣本外 R²（Out-of-Sample R²）'
    if clipped: title += '\n（NN1~NN4、OLS 數值過低已截斷至 -5 顯示）'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    fig.savefig('fig5_oos_r2.png', bbox_inches='tight')
    plt.close()
    print('已儲存 fig5_oos_r2.png')

# ================================================================
# 圖6：各組數下平均勝率比較（折線圖）
# ================================================================
def plot_winrate_by_g():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    g_sorted = sorted(GROUP_SIZES)

    for model, color in zip(MODEL_ORDER, COLORS):
        sub = perf_df[perf_df['model'] == model].set_index('g')
        vals = [sub.loc[g, 'win_rate'] if g in sub.index else np.nan for g in g_sorted]
        ax.plot(range(len(g_sorted)), vals, marker='o', markersize=4,
                label=model, color=color, linewidth=1.5)

    ax.axhline(0.5, color='gray', linewidth=1, linestyle='--', alpha=0.7, label='50% 基準線')
    ax.set_xticks(range(len(g_sorted)))
    ax.set_xticklabels([str(g) for g in g_sorted], fontsize=9)
    ax.set_xlabel('每組股票數 (g)', fontsize=11)
    ax.set_ylabel('勝率', fontsize=11)
    ax.set_ylim(0.38, 0.68)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc='upper right', fontsize=8.5, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title('各組數下勝率比較（Win Rate by Group Size）', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig('fig6_winrate_lines.png', bbox_inches='tight')
    plt.close()
    print('已儲存 fig6_winrate_lines.png')

# ================================================================
# 執行全部
# ================================================================
if __name__ == '__main__':
    print('=== 產生視覺化圖表 ===\n')
    plot_ic_series()
    plot_heatmap_winrate()
    plot_heatmap_sharpe()
    plot_cumret_curves()
    plot_r2()
    plot_winrate_by_g()
    print('\n全部完成！共產出 6 張圖。')
