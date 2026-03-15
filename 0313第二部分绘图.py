import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns
import libpysal
from esda.moran import Moran_Local
from splot.esda import lisa_cluster
import warnings

warnings.filterwarnings('ignore')

# =============================
# 0. 全局配置与路径体系
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='ticks', font='SimHei')

BASE = r'D:\project\城乡\data\pocesseddata'
OUTPUT_DIR = BASE + r'\0313_Section2_Full_Panels'
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
    'COUNTY': BASE + r'\GBAboundary\区县\区县.shp',
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

    T_between = 0
    T_within = 0
    unique_groups = np.unique(g_v)

    for g in unique_groups:
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
# 2. 数据加载与空间尺度拓扑对齐
# =============================
print('>>> [1/5] 加载多维栅格与矢量体系...')
with rasterio.open(RASTERS['POP']) as src:
    meta = src.meta
    ref_shape = (meta['height'], meta['width'])
    transform, crs = meta['transform'], meta['crs']
    pop_arr = src.read(1).astype(np.float32)
    # 提取真实地理边界(Extent)以修复 imshow 坐标错位
    xmin, ymax = transform[2], transform[5]
    xmax = xmin + transform[0] * meta['width']
    ymin = ymax + transform[4] * meta['height']
    raster_extent = [xmin, xmax, ymin, ymax]

with rasterio.open(RASTERS['GDP']) as src: gdp_arr = src.read(1).astype(np.float32)
with rasterio.open(RASTERS['LCZ']) as src: lcz_arr = src.read(1).astype(np.int32)
with rasterio.open(RASTERS['SMOD']) as src: smod_arr = src.read(1).astype(np.int32)
with rasterio.open(RASTERS['GRAD']) as src: grad_arr = src.read(1).astype(np.int32)

gdf_city = gpd.read_file(VECTORS['CITY']).to_crs(crs)
gdf_county = gpd.read_file(VECTORS['COUNTY']).to_crs(crs)
gdf_town = gpd.read_file(VECTORS['TOWN']).to_crs(crs)

city_arr = create_zone_mask(gdf_city, 'name', ref_shape, transform)
county_arr = create_zone_mask(gdf_county, 'name', ref_shape, transform)
town_arr = create_zone_mask(gdf_town, '乡', ref_shape, transform)

# =============================
# 3. 嵌套极化测度与多维泰尔分解
# =============================
print('>>> [2/5] 运行多维泰尔嵌套分解引擎与基尼统计...')
global_gini = weighted_gini(gdp_arr, pop_arr)

# 纵向行政主线分解
_, city_Tb, city_Tw = theil_decomposition(gdp_arr, pop_arr, city_arr)
_, county_Tb, county_Tw = theil_decomposition(gdp_arr, pop_arr, county_arr)
_, town_Tb, town_Tw = theil_decomposition(gdp_arr, pop_arr, town_arr)

# 横向属性主线分解
_, smod_Tb, smod_Tw = theil_decomposition(gdp_arr, pop_arr, smod_arr)
_, lcz_Tb, lcz_Tw = theil_decomposition(gdp_arr, pop_arr, lcz_arr)
_, grad_Tb, grad_Tw = theil_decomposition(gdp_arr, pop_arr, grad_arr)

# 局部基尼测度与映射映射
gini_records = []
town_gini_map = {}
county_gini_map = {}

for cid, cname in enumerate(gdf_city['name'].dropna().unique(), 1):
    g = weighted_gini(gdp_arr[city_arr == cid], pop_arr[city_arr == cid])
    gini_records.append({'Scale': '1_City', 'Unit': cname, 'Gini': g})

for cid, cname in enumerate(gdf_county['name'].dropna().unique(), 1):
    g = weighted_gini(gdp_arr[county_arr == cid], pop_arr[county_arr == cid])
    gini_records.append({'Scale': '2_County', 'Unit': cname, 'Gini': g})
    county_gini_map[cname] = g

for cid, cname in enumerate(gdf_town['乡'].dropna().unique(), 1):
    g = weighted_gini(gdp_arr[town_arr == cid], pop_arr[town_arr == cid])
    gini_records.append({'Scale': '3_Town', 'Unit': cname, 'Gini': g})
    town_gini_map[cname] = g

df_gini = pd.DataFrame(gini_records).dropna()

# 为镇街级数据赋予所属城市（用于后续小提琴图）
gdf_town_pt = gdf_town.copy()
gdf_town_pt.geometry = gdf_town.centroid
joined_town_city = gpd.sjoin(gdf_town_pt, gdf_city[['name', 'geometry']], how='left', predicate='intersects')
town_to_city = dict(zip(joined_town_city['乡'], joined_town_city['name']))

# =============================
# 4. 渲染图表一：宏观基线与全域多维分解
# =============================
print('>>> [3/5] 渲染图表一: 宏观基线与六维泰尔分解图版...')
fig1 = plt.figure(figsize=(22, 10))
gs1 = gridspec.GridSpec(2, 3, figure=fig1, width_ratios=[1.2, 1, 1])

ax1a = fig1.add_subplot(gs1[:, 0])
density = np.full_like(gdp_arr, np.nan, dtype=np.float32)
np.divide(gdp_arr, pop_arr, out=density, where=(pop_arr > 0))

# 注入 extent 修复矢量与栅格的坐标系叠加错位问题
im = ax1a.imshow(density, cmap='magma', extent=raster_extent,
                 norm=LogNorm(vmin=0.1, vmax=np.nanpercentile(density, 99)))
gdf_city.plot(ax=ax1a, facecolor='none', edgecolor='white', linewidth=1.0, alpha=0.8)
ax1a.set_title('(A) 100m 分辨率全域经济密度基线 (消除行政均值化掩盖)', fontsize=15)
ax1a.axis('off')
plt.colorbar(im, ax=ax1a, shrink=0.5, label='GDP Density (Log Scale)')

ax1b = fig1.add_subplot(gs1[0, 1:])
sns.boxplot(data=df_gini, x='Scale', y='Gini', color='white', width=0.4, ax=ax1b)
sns.stripplot(data=df_gini, x='Scale', y='Gini', hue='Scale', alpha=0.6, jitter=True, size=5, palette='viridis',
              legend=False, ax=ax1b)
ax1b.axhline(global_gini, color='red', linestyle='--', label=f'Global Raster Gini = {global_gini:.3f}')
ax1b.set_title('(B) 空间尺度下沉与行政均值化掩盖效应', fontsize=14)
ax1b.legend()

# 扩维后的泰尔分解条形图
ax1c = fig1.add_subplot(gs1[1, 1:])
scales = ['City (城市级)', 'County (区县级)', 'Town (镇街级)', 'SMOD (城乡功能)', 'GRAD (距离梯度)', 'LCZ (物理形态)']
tb_raw = [city_Tb, county_Tb, town_Tb, smod_Tb, grad_Tb, lcz_Tb]
tw_raw = [city_Tw, county_Tw, town_Tw, smod_Tw, grad_Tw, lcz_Tw]

tb_pct = [b / (b + w) * 100 for b, w in zip(tb_raw, tw_raw)]
tw_pct = [w / (b + w) * 100 for b, w in zip(tb_raw, tw_raw)]

ax1c.barh(scales, tb_pct, color='#3b4994', label='Between-group (组间差异)')
ax1c.barh(scales, tw_pct, left=tb_pct, color='#a5add3', label='Within-group (组内极化)')
ax1c.set_xlim(0, 100)
ax1c.set_title('(C) 大湾区总体不平等的多维泰尔分解解释率 (%)', fontsize=14)
ax1c.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig1_Global_Decomposition.png'), dpi=300)
plt.close()

# =============================
# 5. 渲染图表二：中观城市内部结构分异
# =============================
print('>>> [4/5] 渲染图表二: 城市内部区县/镇街级差映射与统计...')
fig2 = plt.figure(figsize=(22, 10))
gs2 = gridspec.GridSpec(1, 2, figure=fig2, width_ratios=[1.2, 1])

# 子图 A: 镇街基尼系数的小提琴图分布
ax2a = fig2.add_subplot(gs2[0])
df_towns = df_gini[df_gini['Scale'] == '3_Town'].copy()
df_towns['Parent_City'] = df_towns['Unit'].map(town_to_city)
df_towns = df_towns.dropna(subset=['Parent_City'])

# 按各市内部极化中位数排序
city_order = df_towns.groupby('Parent_City')['Gini'].median().sort_values(ascending=False).index
sns.violinplot(data=df_towns, x='Parent_City', y='Gini', order=city_order, inner='quartile', palette='muted',
               scale='width', ax=ax2a)
sns.stripplot(data=df_towns, x='Parent_City', y='Gini', order=city_order, color='black', alpha=0.3, size=3, jitter=True,
              ax=ax2a)
ax2a.set_title('(A) 大湾区各城市内部基层极化水平的偏态分布形态', fontsize=15)
ax2a.set_xlabel('所属地级市 (按极化中位数降序排序)', fontsize=12)
ax2a.set_ylabel('镇街级局部基尼系数', fontsize=12)
ax2a.tick_params(axis='x', rotation=30)

# 子图 B: 区县级基尼系数空间制图
ax2b = fig2.add_subplot(gs2[1])
gdf_county['Gini'] = gdf_county['name'].map(county_gini_map)
gdf_city.plot(ax=ax2b, color='#f0f0f0', edgecolor='white', linewidth=1)  # 底图防空洞
gdf_county.plot(column='Gini', cmap='YlOrRd', ax=ax2b, legend=True,
                legend_kwds={'label': 'County-level Gini', 'shrink': 0.7},
                edgecolor='#666666', linewidth=0.5)
gdf_city.plot(ax=ax2b, facecolor='none', edgecolor='black', linewidth=1.2)  # 城市主边界
ax2b.set_title('(B) 中观区县级单元资源配置的空间极化异质合格局', fontsize=15)
ax2b.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig2_Meso_Heterogeneity.png'), dpi=300)
plt.close()

# =============================
# 6. 渲染图表三：基层微观剥夺与本底溯源
# =============================
print('>>> [5/5] 渲染图表三: 镇街级 LISA 自相关与微观物理剖面...')
fig3 = plt.figure(figsize=(22, 12))
gs3 = gridspec.GridSpec(1, 2, figure=fig3, width_ratios=[1.2, 1])

ax3a = fig3.add_subplot(gs3[0])
gdf_town['Gini'] = gdf_town['乡'].map(town_gini_map)
gdf_valid = gdf_town.dropna(subset=['Gini']).reset_index(drop=True)

# KNN 防止孤岛导致权重断裂
w = libpysal.weights.KNN.from_dataframe(gdf_valid, k=5, use_index=True)
w.transform = 'r'
moran_loc = Moran_Local(gdf_valid['Gini'], w)

gdf_city.plot(ax=ax3a, color='#f0f0f0', edgecolor='none', zorder=1)
lisa_cluster(moran_loc, gdf_valid, p=0.05, ax=ax3a, legend=True,
             legend_kwds={'loc': 'lower right', 'fontsize': 12}, zorder=2)
gdf_city.plot(ax=ax3a, facecolor='none', edgecolor='black', linewidth=0.5, zorder=3)
ax3a.set_title('(A) 基层治理单元极化热点空间集聚格局 (Local Moran\'s I)', fontsize=16)
ax3a.axis('off')

ax3b = fig3.add_subplot(gs3[1])
hh_mask = (moran_loc.q == 1) & (moran_loc.p_sim < 0.05)
if hh_mask.sum() > 0:
    typical_town_idx = hh_mask.argmax()
    typical_town_geom = gdf_valid.iloc[typical_town_idx].geometry
    town_name = gdf_valid.iloc[typical_town_idx]['乡']

    with rasterio.open(RASTERS['LCZ']) as dataset:
        out_image, _ = rio_mask(dataset, shapes=[typical_town_geom], crop=True)

    out_image = out_image[0].astype(np.float32)
    out_image[out_image < 1] = np.nan

    cmap_lcz = plt.cm.get_cmap('tab20', 17)
    im3b = ax3b.imshow(out_image, cmap=cmap_lcz, vmin=1, vmax=17)
    ax3b.set_title(f'(B) 典型极化镇街内部的 LCZ 物理空间异质性剖面\n({town_name})', fontsize=16)
    ax3b.axis('off')
    plt.colorbar(im3b, ax=ax3b, label='LCZ Types', shrink=0.7, ticks=range(1, 18))
else:
    ax3b.text(0.5, 0.5, "未检测到显著的 HH 极化区域", ha='center', va='center', fontsize=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Micro_Polarization_LCZ.png'), dpi=300)
plt.close()

print(f'\n>>> 全部模块执行完毕。图版 1、2、3 已输出至: {OUTPUT_DIR}')