import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

# =============================
# 0. 全局配置与顶刊视觉标准设定
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='ticks', font='SimHei')

BASE = r'D:\project\城乡\data\pocesseddata'
OUTPUT_DIR = BASE + r'\0313_Section3_Full_Panels'
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


# =============================
# 1. 核心统计学与空间代数引擎
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
    valid = (y > 1e-6) & (w > 0) & (~np.isnan(y)) & (~np.isnan(w)) & (group_ids > 0)
    y_v, w_v, g_v = y[valid], w[valid], group_ids[valid]
    if len(y_v) == 0: return np.nan, np.nan, np.nan
    Y_total, P_total = np.sum(y_v), np.sum(w_v)
    if Y_total == 0 or P_total == 0: return np.nan, np.nan, np.nan

    global_theil = np.sum((y_v / Y_total) * np.log((y_v / Y_total) / (w_v / P_total)))
    T_between, T_within = 0, 0

    for g in np.unique(g_v):
        zone_mask = (g_v == g)
        Y_g, P_g = np.sum(y_v[zone_mask]), np.sum(w_v[zone_mask])
        if Y_g <= 0 or P_g <= 0: continue
        T_between += (Y_g / Y_total) * np.log((Y_g / Y_total) / (P_g / P_total))
        y_g_arr, w_g_arr = y_v[zone_mask], w_v[zone_mask]
        t_g = np.sum((y_g_arr / Y_g) * np.log((y_g_arr / Y_g) / (w_g_arr / P_g)))
        T_within += (Y_g / Y_total) * t_g

    return global_theil, T_between, T_within


def create_zone_mask(gdf, field, ref_shape, transform):
    unique_vals = gdf[field].dropna().unique()
    val_to_id = {val: i + 1 for i, val in enumerate(unique_vals)}
    shapes = [(geom, val_to_id[val]) for geom, val in zip(gdf.geometry, gdf[field]) if pd.notnull(val)]
    return rasterize(shapes=shapes, out_shape=ref_shape, transform=transform, fill=0, dtype=rasterio.int32)


# =============================
# 2. 数据加载与基础处理
# =============================
print('>>> [1/5] 加载多维空间属性栅格...')
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
city_arr = create_zone_mask(gdf_city, 'name', ref_shape, transform)
town_arr = create_zone_mask(gdf_town, '乡', ref_shape, transform)

# SMOD 类型映射字典 (假设11-30标准分类)
smod_labels = {30: 'Urban Centre', 23: 'Dense Urban', 22: 'Semi-dense Urban',
               21: 'Suburban', 13: 'Rural Cluster', 12: 'Low-density Rural', 11: 'Very Low-density'}

# =============================
# 3. 渲染图表四：城乡定居点(SMOD)分异
# =============================
print('>>> [2/5] 计算与渲染图表四: SMOD 结构分异...')
# 1. 计算局部镇街-SMOD斑块的基尼系数
smod_gini_records = []
for t_id in np.unique(town_arr)[1:]:
    t_mask = (town_arr == t_id)
    for s_id in np.unique(smod_arr[t_mask]):
        if s_id <= 0: continue
        patch_mask = t_mask & (smod_arr == s_id)
        g = weighted_gini(gdp_arr[patch_mask], pop_arr[patch_mask])
        if not np.isnan(g):
            smod_gini_records.append({'SMOD': s_id, 'Gini': g})

df_smod_gini = pd.DataFrame(smod_gini_records)
df_smod_gini['SMOD_Label'] = df_smod_gini['SMOD'].map(smod_labels).fillna(df_smod_gini['SMOD'].astype(str))

# 2. SMOD 内部泰尔分解矩阵
smod_theil_res = []
valid_smods = [s for s in smod_labels.keys() if s in np.unique(smod_arr)]
for s_id in valid_smods:
    s_mask = (smod_arr == s_id)
    y, w = gdp_arr[s_mask], pop_arr[s_mask]
    if len(y) == 0: continue

    _, tb_city, tw_city = theil_decomposition(y, w, city_arr[s_mask])
    _, tb_grad, tw_grad = theil_decomposition(y, w, grad_arr[s_mask])
    _, tb_lcz, tw_lcz = theil_decomposition(y, w, lcz_arr[s_mask])

    # 取三者中最大可解析范围的总泰尔作为基准
    total_T = max(tb_city + tw_city, tb_grad + tw_grad, tb_lcz + tw_lcz)
    if total_T > 0:
        smod_theil_res.append({
            'SMOD_Label': smod_labels[s_id],
            'Admin (City)': tb_city / total_T * 100,
            'Location (Grad)': tb_grad / total_T * 100,
            'Physical (LCZ)': tb_lcz / total_T * 100,
            'Unexplained Within': 100 - ((tb_city + tb_grad + tb_lcz) / total_T * 100)  # 简化表示
        })

df_smod_theil = pd.DataFrame(smod_theil_res).set_index('SMOD_Label')

# 绘图
fig4 = plt.figure(figsize=(20, 9))
gs4 = gridspec.GridSpec(1, 2, figure=fig4, width_ratios=[1.2, 1])

ax4a = fig4.add_subplot(gs4[0])
order_smod = list(smod_labels.values())
sns.boxplot(data=df_smod_gini, x='SMOD_Label', y='Gini', order=order_smod, width=0.5, ax=ax4a, palette='Set2')
sns.stripplot(data=df_smod_gini, x='SMOD_Label', y='Gini', order=order_smod, color='black', alpha=0.15, jitter=True,
              size=3, ax=ax4a)
ax4a.set_title('(A) 不同城乡功能区(SMOD)内部极化分布特征', fontsize=15)
ax4a.tick_params(axis='x', rotation=25)

ax4b = fig4.add_subplot(gs4[1])
colors = ['#4daf4a', '#377eb8', '#e41a1c', '#cccccc']
df_smod_theil.reindex(order_smod).dropna().plot(kind='barh', stacked=True, color=colors, ax=ax4b, edgecolor='white')
ax4b.set_title('(B) SMOD内部差异的多维宏观驱动机制 (Theil分解解释率 %)', fontsize=15)
ax4b.set_xlabel('Contribution Rate (%)')
ax4b.legend(title='Drivers', loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4_SMOD_Nesting.png'), dpi=300)
plt.close()

# =============================
# 4. 渲染图表五：梯度演化与物理肌理叠加
# =============================
print('>>> [3/5] 计算与渲染图表五: 梯度圈层衰减分析 (Twin-axis)...')
unique_grads = np.sort(np.unique(grad_arr[grad_arr > 0]))
grad_stats = []

for g in unique_grads:
    mask = (grad_arr == g)
    y, w = gdp_arr[mask], pop_arr[mask]
    gini_g = weighted_gini(y, w)

    # LCZ 面积占比计算
    lcz_in_g = lcz_arr[mask]
    total_valid = np.sum(lcz_in_g > 0)
    lcz_props = {f'LCZ_{i}': np.sum(lcz_in_g == i) / total_valid if total_valid > 0 else 0 for i in range(1, 18)}

    # 泰尔分解 (按行政城市和SMOD功能)
    _, tb_city, tw_city = theil_decomposition(y, w, city_arr[mask])
    _, tb_smod, tw_smod = theil_decomposition(y, w, smod_arr[mask])
    t_tot = tb_city + tw_city if (tb_city + tw_city) > 0 else 1

    record = {'Gradient': g, 'Gini': gini_g,
              'Tb_City_Pct': tb_city / t_tot * 100, 'Tb_SMOD_Pct': tb_smod / t_tot * 100}
    record.update(lcz_props)
    grad_stats.append(record)

df_grad = pd.DataFrame(grad_stats).dropna(subset=['Gini'])

fig5 = plt.figure(figsize=(22, 10))
gs5 = gridspec.GridSpec(1, 2, figure=fig5, width_ratios=[1.3, 1])

# 子图 A: 梯度折线叠加 LCZ 堆叠面积
ax5a_lcz = fig5.add_subplot(gs5[0])
x_grad = df_grad['Gradient'].values
lcz_cols = [f'LCZ_{i}' for i in range(1, 18)]
y_lcz_stack = np.vstack([df_grad[col].values for col in lcz_cols])

# LCZ 色带
cmap_lcz = plt.cm.get_cmap('tab20', 17)
ax5a_lcz.stackplot(x_grad, y_lcz_stack, labels=[f'LCZ {i}' for i in range(1, 18)],
                   colors=[cmap_lcz(i) for i in range(17)], alpha=0.5)
ax5a_lcz.set_ylabel('LCZ 物理形态覆盖占比 (%)', fontsize=12)
ax5a_lcz.set_xlabel('距核心建成区环境梯度 (Rings/Distance)', fontsize=12)

# 双Y轴绘制基尼折线
ax5a_gini = ax5a_lcz.twinx()
y_gini = df_grad['Gini'].values
# 动态窗口平滑以防过拟合
window = min(len(y_gini) // 5 * 2 + 1, 31)
if window > 3:
    y_gini_smooth = savgol_filter(y_gini, window_length=window, polyorder=3)
    ax5a_gini.plot(x_grad, y_gini_smooth, color='black', linewidth=3, zorder=10, label='Smoothed Gini')
ax5a_gini.scatter(x_grad, y_gini, color='red', s=10, alpha=0.4, label='Raw Gini')
ax5a_gini.set_ylabel('总体基尼系数测度', fontsize=12)

ax5a_lcz.set_title('(A) 空间极化的梯度圈层衰减与底层三维物理肌理演替', fontsize=15)
# 简化图例以防遮挡
handles1, labels1 = ax5a_lcz.get_legend_handles_labels()
handles2, labels2 = ax5a_gini.get_legend_handles_labels()
ax5a_lcz.legend(handles1[::2] + handles2, labels1[::2] + labels2, loc='upper right', bbox_to_anchor=(1.25, 1),
                fontsize=9)

# 子图 B: 梯度上的泰尔指数演化
ax5b = fig5.add_subplot(gs5[1])
y_theil_stack = np.vstack([df_grad['Tb_City_Pct'].values, df_grad['Tb_SMOD_Pct'].values,
                           100 - (df_grad['Tb_City_Pct'].values + df_grad['Tb_SMOD_Pct'].values)])
ax5b.stackplot(x_grad, y_theil_stack, labels=['Admin City (行政壁垒)', 'SMOD (功能分异)', 'Within (微观极化)'],
               colors=['#3b4994', '#5ac8c8', '#e8e8e8'])
ax5b.set_title('(B) 距离梯度约束下的极化驱动力动态演变', fontsize=15)
ax5b.set_xlabel('建成区环境梯度', fontsize=12)
ax5b.set_ylabel('Theil Decomposition (%)', fontsize=12)
ax5b.legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig5_Gradient_Decay.png'), dpi=300)
plt.close()

# =============================
# 5. 渲染图表六：LCZ 底向均质性分异
# =============================
print('>>> [4/5] 计算与渲染图表六: LCZ 物理本底山峦图与热力图...')

lcz_gini_records = []
for t_id in np.unique(town_arr)[1:]:
    t_mask = (town_arr == t_id)
    for l_id in range(1, 18):
        patch_mask = t_mask & (lcz_arr == l_id)
        g = weighted_gini(gdp_arr[patch_mask], pop_arr[patch_mask])
        if not np.isnan(g):
            lcz_gini_records.append({'LCZ': l_id, 'Gini': g})
df_lcz_gini = pd.DataFrame(lcz_gini_records)

# 建立 LCZ 泰尔热力图矩阵
lcz_matrix = np.zeros((17, 3))
for l_id in range(1, 18):
    l_mask = (lcz_arr == l_id)
    y, w = gdp_arr[l_mask], pop_arr[l_mask]
    if len(y) > 0:
        _, tb_city, tw_city = theil_decomposition(y, w, city_arr[l_mask])
        _, tb_smod, tw_smod = theil_decomposition(y, w, smod_arr[l_mask])
        _, tb_grad, tw_grad = theil_decomposition(y, w, grad_arr[l_mask])
        tot = max(tb_city + tw_city, 1e-6)
        lcz_matrix[l_id - 1] = [tb_city / tot * 100, tb_smod / tot * 100, tb_grad / tot * 100]

fig6 = plt.figure(figsize=(20, 10))
gs6 = gridspec.GridSpec(1, 2, figure=fig6, width_ratios=[1, 1.2])

# 子图 A: 手工构建高规格山峦图 (Ridge Plot)
ax6a = fig6.add_subplot(gs6[0])
lcz_types = range(1, 18)
overlap = 1.5  # 山峦重叠度
x_eval = np.linspace(0, 1, 200)

for i, l_id in enumerate(lcz_types):
    data = df_lcz_gini[df_lcz_gini['LCZ'] == l_id]['Gini'].values
    if len(data) > 5:
        kde = gaussian_kde(data, bw_method=0.3)
        y_eval = kde(x_eval)
        # 归一化并施加垂直偏移
        y_eval = y_eval / np.max(y_eval) * overlap
        ax6a.fill_between(x_eval, i, y_eval + i, facecolor=cmap_lcz(l_id - 1), alpha=0.7, zorder=17 - i)
        ax6a.plot(x_eval, y_eval + i, color='white', lw=1, zorder=17 - i + 0.5)

ax6a.set_yticks(np.arange(17))
ax6a.set_yticklabels([f'LCZ {i}' for i in range(1, 18)])
ax6a.set_ylim(-1, 18)
ax6a.set_xlim(0, 1)
ax6a.set_xlabel('基层局部基尼系数', fontsize=12)
ax6a.set_title('(A) 不同物理形态(LCZ)自带的社会经济极化概率分布 (核密度)', fontsize=15)

# 子图 B: 热力图矩阵
ax6b = fig6.add_subplot(gs6[1])
sns.heatmap(lcz_matrix, cmap='YlGnBu', annot=True, fmt=".1f",
            xticklabels=['Admin (City)', 'Function (SMOD)', 'Location (GRAD)'],
            yticklabels=[f'LCZ {i}' for i in range(1, 18)],
            cbar_kws={'label': 'Between-group Theil Contribution (%)'}, ax=ax6b)
ax6b.set_title('(B) 同类物理景观内部极化的宏观空间约束机制诊断', fontsize=15)
ax6b.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig6_LCZ_Matrix.png'), dpi=300)
plt.close()

print(f'\n>>> 第三模块全部复杂空间图版编译完成！输出至: {OUTPUT_DIR}')