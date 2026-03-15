import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# =============================
# 0. 全局配置与色彩系统
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='ticks', font='SimHei')

BASE = r'D:\project\城乡\data\pocesseddata'
OUTPUT_DIR = BASE + r'\0313_Section3_Rigorous'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

RASTERS = {
    'POP': BASE + r'\0313aligned_output\Master_POP_100m.tif',
    'GDP': BASE + r'\0313aligned_output\Aligned_100m_GDP.tif',
    'LCZ': BASE + r'\0313aligned_output\Aligned_100m_LCZ.tif',
    'SMOD': BASE + r'\0313aligned_output\Aligned_100m_SMOD.tif',
    'GRAD': BASE + r'\0313aligned_output\Aligned_100m_GRAD.tif'
}

VECTORS = {
    'CITY': BASE + r'\GBAboundary\boundarymollweide.shp',
    'TOWN': r'D:\project\城乡\data\区划数据\wx_GBAwithoutHKMC\GBAwithoutHM.shp'
}

# 严谨的色系配对 (深色代表 Tb组间解释率，浅色代表 Tw组内极化)
COLOR_MAP = {
    'City': ('#3b4994', '#c2c8e6'),  # 蓝色系 (行政)
    'Grad': ('#d95f02', '#fed6b4'),  # 橙色系 (区位)
    'LCZ': ('#4daf4a', '#ccebc5'),  # 绿色系 (物理形态)
    'SMOD': ('#e41a1c', '#fbb4b9')  # 红色系 (功能属性)
}


# =============================
# 1. 核心数学统计引擎
# =============================
def weighted_gini(y, w):
    valid = (y >= 0) & (w > 0) & (~np.isnan(y)) & (~np.isnan(w))
    y, w = y[valid], w[valid]
    if len(y) == 0 or np.sum(w) == 0: return np.nan
    idx = np.argsort(y)
    y, w = y[idx], w[idx]
    cum_w, cum_yw = np.cumsum(w), np.cumsum(y * w)
    X, Y = cum_w / cum_w[-1], cum_yw / cum_yw[-1]
    X, Y = np.insert(X, 0, 0.0), np.insert(Y, 0, 0.0)
    B = np.sum((Y[1:] + Y[:-1]) / 2.0 * (X[1:] - X[:-1]))
    return max(0.0, float(1.0 - 2.0 * B))


def theil_decomposition(y, w, group_ids):
    """返回：总体泰尔，组间泰尔，组内泰尔"""
    valid = (y > 1e-6) & (w > 0) & (~np.isnan(y)) & (~np.isnan(w)) & (group_ids > 0)
    y_v, w_v, g_v = y[valid], w[valid], group_ids[valid]
    if len(y_v) == 0: return 0, 0, 0
    Y_total, P_total = np.sum(y_v), np.sum(w_v)
    if Y_total <= 0 or P_total <= 0: return 0, 0, 0

    global_theil = np.sum((y_v / Y_total) * np.log((y_v / Y_total) / (w_v / P_total)))
    T_between = 0

    for g in np.unique(g_v):
        mask = (g_v == g)
        Y_g, P_g = np.sum(y_v[mask]), np.sum(w_v[mask])
        if Y_g <= 0 or P_g <= 0: continue
        T_between += (Y_g / Y_total) * np.log((Y_g / Y_total) / (P_g / P_total))

    T_within = global_theil - T_between
    return global_theil, T_between, T_within


def plot_grouped_stacked_bars(ax, df, x_labels, drivers, title):
    """数学严谨版：簇状堆叠柱形图渲染器 (用于类别型变量如SMOD和LCZ)"""
    x = np.arange(len(x_labels))
    n_groups = len(drivers)
    width = 0.8 / n_groups

    for i, driver in enumerate(drivers):
        offset = (i - n_groups / 2 + 0.5) * width
        tb_vals = df[f'Tb_{driver}']
        tw_vals = df[f'Tw_{driver}']
        color_tb, color_tw = COLOR_MAP[driver]

        ax.bar(x + offset, tb_vals, width, label=f'Explained by {driver} (Tb)', color=color_tb, edgecolor='white',
               linewidth=0.5)
        ax.bar(x + offset, tw_vals, width, bottom=tb_vals, label=f'Unexplained (Tw)', color=color_tw, edgecolor='white',
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('泰尔指数总差异占比 (100%)', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)


def create_zone_mask(gdf, field, ref_shape, transform):
    unique_vals = gdf[field].dropna().unique()
    val_to_id = {val: i + 1 for i, val in enumerate(unique_vals)}
    id_to_val = {i + 1: val for i, val in enumerate(unique_vals)}
    shapes = [(geom, val_to_id[val]) for geom, val in zip(gdf.geometry, gdf[field]) if pd.notnull(val)]
    return rasterize(shapes=shapes, out_shape=ref_shape, transform=transform, fill=0, dtype=rasterio.int32), id_to_val


# =============================
# 2. 数据加载与预处理
# =============================
print('>>> [1/4] 加载空间数据与掩模...')
with rasterio.open(RASTERS['POP']) as src:
    meta = src.meta
    ref_shape = (meta['height'], meta['width'])
    transform = meta['transform']
    pop_arr = src.read(1).astype(np.float32)

with rasterio.open(RASTERS['GDP']) as src: gdp_arr = src.read(1).astype(np.float32)
with rasterio.open(RASTERS['LCZ']) as src: lcz_arr = src.read(1).astype(np.int32)
with rasterio.open(RASTERS['SMOD']) as src: smod_arr = src.read(1).astype(np.int32)
with rasterio.open(RASTERS['GRAD']) as src: grad_arr = src.read(1).astype(np.int32)

gdf_city = gpd.read_file(VECTORS['CITY']).to_crs(meta['crs'])
gdf_town = gpd.read_file(VECTORS['TOWN']).to_crs(meta['crs'])
city_arr, city_dict = create_zone_mask(gdf_city, 'name', ref_shape, transform)
town_arr, _ = create_zone_mask(gdf_town, '乡', ref_shape, transform)

# =============================
# 3. 第 3.1 节：SMOD 结构分异 (图表 4)
# =============================
print('>>> [2/4] 生成 3.1 节 SMOD 严谨嵌套图版...')
smod_gini_list, smod_theil_list = [], []
smod_types = [11, 12, 13, 21, 22, 23, 30]

city_smod_matrix = pd.DataFrame(index=city_dict.values(), columns=smod_types)
for cid, cname in city_dict.items():
    c_mask = (city_arr == cid)
    for s_id in smod_types:
        p_mask = c_mask & (smod_arr == s_id)
        city_smod_matrix.loc[cname, s_id] = weighted_gini(gdp_arr[p_mask], pop_arr[p_mask])

for t_id in np.unique(town_arr)[1:]:
    t_mask = (town_arr == t_id)
    for s_id in smod_types:
        p_mask = t_mask & (smod_arr == s_id)
        g = weighted_gini(gdp_arr[p_mask], pop_arr[p_mask])
        if not np.isnan(g): smod_gini_list.append({'SMOD': s_id, 'Gini': g})

for s_id in smod_types:
    mask = (smod_arr == s_id)
    y, w = gdp_arr[mask], pop_arr[mask]
    if len(y) > 0:
        g_T, tb_city, _ = theil_decomposition(y, w, city_arr[mask])
        _, tb_grad, _ = theil_decomposition(y, w, grad_arr[mask])
        _, tb_lcz, _ = theil_decomposition(y, w, lcz_arr[mask])

        tot = g_T if g_T > 0 else 1e-6
        smod_theil_list.append({
            'SMOD': str(s_id),
            'Tb_City': tb_city / tot * 100, 'Tw_City': 100 - (tb_city / tot * 100),
            'Tb_Grad': tb_grad / tot * 100, 'Tw_Grad': 100 - (tb_grad / tot * 100),
            'Tb_LCZ': tb_lcz / tot * 100, 'Tw_LCZ': 100 - (tb_lcz / tot * 100)
        })

fig4 = plt.figure(figsize=(22, 14))
gs4 = gridspec.GridSpec(2, 2, figure=fig4, height_ratios=[1.2, 1])

ax4a = fig4.add_subplot(gs4[0, 0])
df_smod_gini = pd.DataFrame(smod_gini_list)
sns.boxplot(data=df_smod_gini, x='SMOD', y='Gini', palette='Set2', ax=ax4a, fliersize=0)
sns.stripplot(data=df_smod_gini, x='SMOD', y='Gini', color='black', alpha=0.3, jitter=True, size=3, ax=ax4a)
ax4a.set_title('(A) 基层镇街单元内各 SMOD 的局部基尼分布基线', fontsize=14)

ax4b = fig4.add_subplot(gs4[0, 1])
sns.heatmap(city_smod_matrix.astype(float), cmap='YlOrRd', annot=True, fmt=".2f", ax=ax4b, linewidths=0.5)
ax4b.set_facecolor('#e0e0e0')
ax4b.set_title('(B) 同类 SMOD 在不同城市的资源极化横向异质性', fontsize=14)

ax4c = fig4.add_subplot(gs4[1, :])
df_smod_theil = pd.DataFrame(smod_theil_list)
plot_grouped_stacked_bars(ax4c, df_smod_theil, df_smod_theil['SMOD'], ['City', 'Grad', 'LCZ'],
                          '(C) 同类SMOD内部总泰尔指数的多维独立解释率 (%)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_3_1_SMOD_Rigorous.png'), dpi=300)
plt.close()

# =============================
# 4. 第 3.2 节：建成区环境梯度 (图表 5 - 全新堆叠面积排版)
# =============================
print('>>> [3/4] 生成 3.2 节 建成区梯度衰减图版 (分离式堆叠面积图)...')
grad_gini_list, grad_theil_list = [], []

# 用于连续折线与面积图的全梯度遍历 (1-50)
continuous_grads = np.sort(np.unique(grad_arr[(grad_arr > 0) & (grad_arr <= 50)]))

for g in continuous_grads:
    mask = (grad_arr == g)
    y, w = gdp_arr[mask], pop_arr[mask]

    g_val = weighted_gini(y, w)
    if not np.isnan(g_val): grad_gini_list.append({'Gradient': g, 'Gini': g_val})

    if len(y) > 0:
        g_T, tb_city, _ = theil_decomposition(y, w, city_arr[mask])
        _, tb_smod, _ = theil_decomposition(y, w, smod_arr[mask])
        _, tb_lcz, _ = theil_decomposition(y, w, lcz_arr[mask])

        tot = g_T if g_T > 0 else 1e-6
        grad_theil_list.append({
            'Gradient': g,
            'Tb_City': tb_city / tot * 100, 'Tw_City': 100 - (tb_city / tot * 100),
            'Tb_SMOD': tb_smod / tot * 100, 'Tw_SMOD': 100 - (tb_smod / tot * 100),
            'Tb_LCZ': tb_lcz / tot * 100, 'Tw_LCZ': 100 - (tb_lcz / tot * 100)
        })

df_g_gini = pd.DataFrame(grad_gini_list)
df_g_theil = pd.DataFrame(grad_theil_list).set_index('Gradient')

# 用于热力图的归并区间 (否则50列太密)
grad_bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50)]
bin_labels = [f'{b[0] + 1}-{b[1]}' for b in grad_bins]
city_grad_matrix = pd.DataFrame(index=city_dict.values(), columns=bin_labels)

for (lower, upper), label in zip(grad_bins, bin_labels):
    mask = (grad_arr > lower) & (grad_arr <= upper)
    for cid, cname in city_dict.items():
        p_mask = mask & (city_arr == cid)
        city_grad_matrix.loc[cname, label] = weighted_gini(gdp_arr[p_mask], pop_arr[p_mask])

# 重新设计 2行3列 的新排版
fig5 = plt.figure(figsize=(24, 12))
# 第一行 2列 (各占一半)，第二行 3列
gs5 = gridspec.GridSpec(2, 6, figure=fig5, height_ratios=[1.2, 1])

ax5a = fig5.add_subplot(gs5[0, :3])
ax5a.plot(df_g_gini['Gradient'], df_g_gini['Gini'], marker='o', color='#333333', linewidth=2, markersize=4)
ax5a.set_title('(A) 圈层衰减：不同梯度上的总体基尼系数连续演化', fontsize=14)
ax5a.set_xlabel('建成区环境梯度 (Gradient)')
ax5a.set_ylabel('总体基尼系数')
ax5a.grid(True, linestyle='--', alpha=0.6)

ax5b = fig5.add_subplot(gs5[0, 3:])
sns.heatmap(city_grad_matrix.astype(float), cmap='YlOrRd', annot=True, fmt=".2f", ax=ax5b, linewidths=0.5)
ax5b.set_facecolor('#e0e0e0')
ax5b.set_title('(B) 城市级圈层衰减异质性 (区域截面热力矩阵)', fontsize=14)


# 底部 3个面积图分别展示 City, SMOD, LCZ
def plot_area_decomp(ax, driver, title_prefix):
    color_tb, color_tw = COLOR_MAP[driver]
    ax.stackplot(df_g_theil.index, df_g_theil[f'Tb_{driver}'], df_g_theil[f'Tw_{driver}'],
                 labels=[f'{driver} 组间 (Tb)', '组内残差 (Tw)'], colors=[color_tb, color_tw], alpha=0.85)
    ax.set_title(title_prefix, fontsize=13)
    ax.set_xlabel('建成区环境梯度')
    ax.set_ylim(0, 100)
    if driver == 'City': ax.set_ylabel('泰尔指数分解占比 (%)')
    ax.legend(loc='lower left', fontsize=10)


ax5c = fig5.add_subplot(gs5[1, :2])
plot_area_decomp(ax5c, 'City', '(C) 独立剥离：[行政落差]在各圈层的极化解释力')

ax5d = fig5.add_subplot(gs5[1, 2:4])
plot_area_decomp(ax5d, 'SMOD', '(D) 独立剥离：[城乡功能]在各圈层的极化解释力')

ax5e = fig5.add_subplot(gs5[1, 4:])
plot_area_decomp(ax5e, 'LCZ', '(E) 独立剥离：[物理形态]在各圈层的极化解释力')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_3_2_Gradient_Rigorous.png'), dpi=300)
plt.close()

# =============================
# 5. 第 3.3 节：LCZ 物理底向分异 (图表 6)
# =============================
print('>>> [4/4] 生成 3.3 节 LCZ 物理形态分异图版...')
lcz_gini_list, lcz_theil_list = [], []
lcz_types = list(range(1, 18))

city_lcz_matrix = pd.DataFrame(index=city_dict.values(), columns=[f'LCZ_{i}' for i in lcz_types])
for cid, cname in city_dict.items():
    c_mask = (city_arr == cid)
    for l_id in lcz_types:
        p_mask = c_mask & (lcz_arr == l_id)
        city_lcz_matrix.loc[cname, f'LCZ_{l_id}'] = weighted_gini(gdp_arr[p_mask], pop_arr[p_mask])

for t_id in np.unique(town_arr)[1:]:
    t_mask = (town_arr == t_id)
    for l_id in lcz_types:
        p_mask = t_mask & (lcz_arr == l_id)
        g = weighted_gini(gdp_arr[p_mask], pop_arr[p_mask])
        if not np.isnan(g): lcz_gini_list.append({'LCZ': str(l_id), 'Gini': g})

for l_id in lcz_types:
    mask = (lcz_arr == l_id)
    y, w = gdp_arr[mask], pop_arr[mask]
    if len(y) > 0:
        g_T, tb_city, _ = theil_decomposition(y, w, city_arr[mask])
        _, tb_smod, _ = theil_decomposition(y, w, smod_arr[mask])
        _, tb_grad, _ = theil_decomposition(y, w, grad_arr[mask])

        tot = g_T if g_T > 0 else 1e-6
        lcz_theil_list.append({
            'LCZ': str(l_id),
            'Tb_City': tb_city / tot * 100, 'Tw_City': 100 - (tb_city / tot * 100),
            'Tb_SMOD': tb_smod / tot * 100, 'Tw_SMOD': 100 - (tb_smod / tot * 100),
            'Tb_Grad': tb_grad / tot * 100, 'Tw_Grad': 100 - (tb_grad / tot * 100)
        })

fig6 = plt.figure(figsize=(24, 15))
gs6 = gridspec.GridSpec(2, 2, figure=fig6, height_ratios=[1.2, 1])

ax6a = fig6.add_subplot(gs6[0, 0])
df_lcz_gini = pd.DataFrame(lcz_gini_list)
sns.boxplot(data=df_lcz_gini, x='LCZ', y='Gini', palette='Spectral',
            order=[str(i) for i in range(1, 18)], ax=ax6a, fliersize=0)
sns.stripplot(data=df_lcz_gini, x='LCZ', y='Gini', color='black', alpha=0.3, jitter=True, size=2,
              order=[str(i) for i in range(1, 18)], ax=ax6a)
ax6a.set_title('(A) 基层镇街内各类 LCZ 物理景观的微观基尼分布基线', fontsize=14)

ax6b = fig6.add_subplot(gs6[0, 1])
sns.heatmap(city_lcz_matrix.astype(float), cmap='YlOrRd', annot=True, fmt=".2f", ax=ax6b, linewidths=0.5,
            annot_kws={'size': 9})
ax6b.set_facecolor('#e0e0e0')
ax6b.set_title('(B) 同类 LCZ 在不同城市的社会空间极化横向异质性', fontsize=14)

ax6c = fig6.add_subplot(gs6[1, :])
df_lcz_theil = pd.DataFrame(lcz_theil_list)
plot_grouped_stacked_bars(ax6c, df_lcz_theil, df_lcz_theil['LCZ'], ['City', 'SMOD', 'Grad'],
                          '(C) 同类物理形态内部总泰尔的多维独立解释率 (%)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_3_3_LCZ_Rigorous.png'), dpi=300)
plt.close()

print(f'\n>>> 数学逻辑绝对严密（包含平滑分离堆叠图）的终极版代码执行完毕！')