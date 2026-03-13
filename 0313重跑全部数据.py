import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置与路径体系
# ==========================================
BASE = r'D:\project\城乡\data\pocesseddata'
INPUT_DIR = os.path.join(BASE, '0313aligned_output')
OUTPUT_DIR = os.path.join(BASE, '0314_GBA_Full_Analysis_Final')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

RASTERS = {
    'POP': os.path.join(INPUT_DIR, 'Master_POP_100m.tif'),
    'GDP': os.path.join(INPUT_DIR, 'Aligned_100m_GDP.tif'),
    'SMOD': os.path.join(INPUT_DIR, 'Aligned_100m_SMOD.tif'),
    'LCZ': os.path.join(INPUT_DIR, 'Aligned_100m_LCZ.tif'),
    'GRAD': os.path.join(INPUT_DIR, 'Aligned_100m_GRAD.tif')
}

VECTORS = {
    'CITY': {'path': os.path.join(BASE, r'GBAboundary\boundarymollweide.shp'), 'field': 'name'},
    'COUNTY': {'path': os.path.join(BASE, r'GBAboundary\区县\区县.shp'), 'field': 'name'},
    'TOWN': {'path': r'D:\project\城乡\data\区划数据\wx_GBAwithoutHKMC\GBAwithoutHM.shp', 'field': '乡'}
}


# ==========================================
# 2. 增强型数学函数 (输出完整泰尔成分)
# ==========================================
def weighted_gini(x, w):
    x, w = np.asarray(x), np.asarray(w)
    mask = (x >= 0) & (w > 0)
    x, w = x[mask], w[mask]
    if len(x) <= 1: return 0.0
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cum_w, cum_xw = np.cumsum(w), np.cumsum(x * w)
    X, Y = cum_w / cum_w[-1], cum_xw / cum_xw[-1]
    X, Y = np.insert(X, 0, 0.0), np.insert(Y, 0, 0.0)
    return 1.0 - 2.0 * np.sum((Y[1:] + Y[:-1]) / 2.0 * (X[1:] - X[:-1]))


def theil_decomposition_full(df, val_col, weight_col, group_col):
    """泰尔指数分解：返回总额、组间、组内及贡献率 [cite: 7, 12, 28]"""
    temp = df[[val_col, weight_col, group_col]].copy()
    temp = temp[(temp[val_col] >= 0) & (temp[weight_col] > 0) & (temp[group_col].notna())]
    if temp.empty or temp[group_col].nunique() <= 1:
        return 0.0, 0.0, 0.0, 0.0

    temp['weighted_val'] = temp[val_col] * temp[weight_col]
    total_val, total_pop = temp['weighted_val'].sum(), temp[weight_col].sum()

    # 总体泰尔
    s_i, p_i = temp['weighted_val'].values / total_val, temp[weight_col].values / total_pop
    total_theil = np.sum(s_i * np.log(s_i / p_i + 1e-12))

    # 组间泰尔
    groups = temp.groupby(group_col).agg(gv=('weighted_val', 'sum'), gp=(weight_col, 'sum'))
    S_j, P_j = groups['gv'].values / total_val, groups['gp'].values / total_pop
    theil_between = np.sum(S_j * np.log(S_j / P_j + 1e-12))

    theil_within = total_theil - theil_between
    contribution = (theil_between / total_theil) if total_theil > 0 else 0
    return total_theil, theil_between, theil_within, contribution


# ==========================================
# 3. 数据处理与物理对齐
# ==========================================
print(">>> [Step 1/3] 加载栅格并同步矢量...")
with rasterio.open(RASTERS['POP']) as src:
    meta, transform, ref_shape = src.meta, src.transform, src.shape
    data_dict = {'POP': src.read(1).ravel()}

for key in ['GDP', 'SMOD', 'LCZ', 'GRAD']:
    with rasterio.open(RASTERS[key]) as src:
        data_dict[key] = src.read(1).ravel()

name_maps = {}
for key, config in VECTORS.items():
    gdf = gpd.read_file(config['path']).to_crs(meta['crs'])
    shapes = ((geom, idx) for idx, geom in zip(gdf.index, gdf.geometry))
    rasterized = rasterize(shapes, out_shape=ref_shape, transform=transform, fill=-1, dtype='int32')
    data_dict[key] = rasterized.ravel()
    name_maps[f'{key}_MAP'] = gdf[config['field']].to_dict()

df_master = pd.DataFrame(data_dict)
df_master = df_master[(df_master['POP'] > 0) & (df_master['GDP'] >= 0)].reset_index(drop=True)
df_master['CITY_NAME'] = df_master['CITY'].map(name_maps.get('CITY_MAP', {}))
df_master['COUNTY_NAME'] = df_master['COUNTY'].map(name_maps.get('COUNTY_MAP', {}))
df_master['TOWN_NAME'] = df_master['TOWN'].map(name_maps.get('TOWN_MAP', {}))

# ==========================================
# 4. 执行研究框架全流程计算 (1.1 - 1.7)
# ==========================================
print(">>> [Step 2/3] 执行全维度计算与分解...")
summary_log = []

# --- 1.1 全域基线 (含泰尔完整成分) ---
g_gini = weighted_gini(df_master['GDP'], df_master['POP'])
summary_log.append({'Unit': 'Pixel', 'Metric': 'Gini', 'Value': g_gini})

for u in ['CITY', 'COUNTY', 'TOWN', 'SMOD', 'GRAD', 'LCZ']:
    # 聚合均值基尼
    agg = df_master.groupby(u).agg({'GDP': 'mean', 'POP': 'sum'}).dropna()
    summary_log.append({'Unit': u, 'Metric': 'Aggregated_Gini', 'Value': weighted_gini(agg['GDP'], agg['POP'])})
    # 泰尔分解
    t_tot, t_bet, t_wit, c_rate = theil_decomposition_full(df_master, 'GDP', 'POP', u)
    summary_log.append({'Unit': u, 'Metric': 'Theil_Total', 'Value': t_tot})
    summary_log.append({'Unit': u, 'Metric': 'Theil_Between', 'Value': t_bet})
    summary_log.append({'Unit': u, 'Metric': 'Theil_Within', 'Value': t_wit})
    summary_log.append({'Unit': u, 'Metric': 'Theil_Contribution_Rate', 'Value': c_rate})


# --- 下沉嵌套分析函数 (增加泰尔成分保存) ---
def run_full_analysis(group_key, sub_dims, out_name):
    print(f"    - 正在处理: {out_name}")
    results = []
    for name, group in df_master.groupby(group_key):
        if len(group) < 5 or pd.isna(name): continue
        results.append({'Group': name, 'Metric': 'Internal_Gini', 'Value': weighted_gini(group['GDP'], group['POP'])})
        for sd in sub_dims:
            t_tot, t_bet, t_wit, c_rate = theil_decomposition_full(group, 'GDP', 'POP', sd)
            results.append({'Group': name, 'Metric': f'Theil_{sd}_Total', 'Value': t_tot})
            results.append({'Group': name, 'Metric': f'Theil_{sd}_Between', 'Value': t_bet})
            results.append({'Group': name, 'Metric': f'Theil_{sd}_Within', 'Value': t_wit})
            results.append({'Group': name, 'Metric': f'Theil_{sd}_Contribution', 'Value': c_rate})
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)


# 1.2 - 1.4 行政下沉 (补全了 1.4 镇街级)
run_full_analysis('CITY_NAME', ['COUNTY', 'TOWN', 'SMOD', 'GRAD', 'LCZ'], '1.2_City_Analysis.csv')
run_full_analysis('COUNTY_NAME', ['TOWN', 'SMOD', 'GRAD', 'LCZ'], '1.3_County_Analysis.csv')
run_full_analysis('TOWN_NAME', ['SMOD', 'GRAD', 'LCZ'], '1.4_Town_Analysis.csv')

# 1.5 - 1.7 属性解构
run_full_analysis('SMOD', ['CITY', 'COUNTY', 'GRAD', 'LCZ'], '1.5_SMOD_Analysis.csv')
run_full_analysis('GRAD', ['CITY', 'SMOD', 'LCZ'], '1.6_Gradient_Analysis.csv')
run_full_analysis('LCZ', ['CITY', 'SMOD', 'GRAD'], '1.7_LCZ_Analysis.csv')

# ==========================================
# 5. 结果导出
# ==========================================
pd.DataFrame(summary_log).to_csv(os.path.join(OUTPUT_DIR, '1.1_Global_Baseline.csv'), index=False)
print(f"\n>>> 计算圆满结束。结果包含泰尔成分，位于: {OUTPUT_DIR}")