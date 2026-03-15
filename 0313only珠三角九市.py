import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# =============================
# 0. 全局配置与参数
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='white', font='SimHei')

BASE = r'D:\project\城乡\data\pocesseddata'
OUTPUT_DIR = BASE + r'\0313GINItestoutput_maps_NoHKMC'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 输入数据路径
RASTERS = {
    'POP': BASE + r'\0312aligned_output\Master_POP_100m.tif',
    'GDP': BASE + r'\0312aligned_output\Aligned_100m_GDP.tif',
    'LCZ': BASE + r'\0312aligned_output\Aligned_100m_LCZ.tif',
    'SMOD': BASE + r'\0312aligned_output\Aligned_100m_SMOD.tif',
    'GRAD': BASE + r'\0312aligned_output\Aligned_100m_GRAD.tif'
}

VECTOR_CITY = BASE + r'\GBAboundary\boundarymollweide.shp'

# =============================
# 1. 空间数据加载与双掩模构建 (剔除港澳)
# =============================
print('>>> [1/4] 启动空间数据读取与珠三角九市掩模初始化...')
arrays = {}
with rasterio.open(RASTERS['POP']) as src:
    meta = src.meta.copy()
    ref_shape = (meta['height'], meta['width'])
    transform_val = meta['transform']
    crs_val = meta['crs']
    arrays['POP'] = src.read(1).astype(np.float32)
    arrays['POP'][arrays['POP'] == src.nodata] = np.nan
    arrays['POP'][arrays['POP'] < 0] = np.nan

with rasterio.open(RASTERS['GDP']) as src: arrays['GDP'] = src.read(1).astype(np.float32)
with rasterio.open(RASTERS['LCZ']) as src: arrays['LCZ'] = src.read(1).astype(np.int32)
with rasterio.open(RASTERS['SMOD']) as src: arrays['SMOD'] = src.read(1).astype(np.int32)
with rasterio.open(RASTERS['GRAD']) as src: arrays['GRAD'] = src.read(1).astype(np.int32)

gdf_city = gpd.read_file(VECTOR_CITY).to_crs(crs_val)

# 剔除香港与澳门特别行政区边界
col_name = 'name' if 'name' in gdf_city.columns else ('NAME' if 'NAME' in gdf_city.columns else None)
if col_name:
    gdf_city = gdf_city[~gdf_city[col_name].str.contains('香港|澳门|Hong Kong|Macao|Macau', na=False, regex=True)]

gdf_city = gdf_city.reset_index(drop=True)

geom_list = [(geom, i + 1) for i, geom in enumerate(gdf_city.geometry)]
city_matrix = rasterize(shapes=geom_list, out_shape=ref_shape, transform=transform_val, fill=0, dtype=rasterio.uint8)

# 构建计算与绘图掩模
valid_calc_mask = (city_matrix > 0) & (arrays['POP'] > 0) & (~np.isnan(arrays['GDP'])) & (arrays['GDP'] >= 0)
geo_plot_mask = (city_matrix > 0) & (~np.isnan(arrays['POP']))

xmin = transform_val[2]
xmax = transform_val[2] + transform_val[0] * meta['width']
ymin = transform_val[5] + transform_val[4] * meta['height']
ymax = transform_val[5]
raster_extent = [xmin, xmax, ymin, ymax]


# 基尼系数计算引擎
def weighted_gini(x, w):
    mask = (x >= 0) & (w > 0) & (~np.isnan(x)) & (~np.isnan(w))
    x, w = x[mask], w[mask]
    if len(x) == 0: return np.nan
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cum_w = np.cumsum(w)
    cum_xw = np.cumsum(x * w)
    X = cum_w / cum_w[-1]
    Y = cum_xw / cum_xw[-1]
    X = np.insert(X, 0, 0.0)
    Y = np.insert(Y, 0, 0.0)
    B = np.sum((Y[1:] + Y[:-1]) / 2.0 * (X[1:] - X[:-1]))
    return float(1.0 - 2.0 * B)


# =============================
# 2. 核心代数函数：全域/嵌套基尼空间映射引擎
# =============================
def generate_gini_maps(class_array, max_class_val=None):
    """
    针对全域的某种分类栅格(SMOD/LCZ/GRAD)，计算并生成两张地图：
    1. map_global: 珠三角九市统筹的基尼系数映射
    2. map_nested: 分城市的嵌套基尼系数映射
    """
    map_global = np.full(ref_shape, np.nan, dtype=np.float32)
    map_nested = np.full(ref_shape, np.nan, dtype=np.float32)

    # 提取有效分类
    unique_classes = np.unique(class_array[valid_calc_mask])
    unique_classes = unique_classes[unique_classes > 0]
    if max_class_val:
        unique_classes = unique_classes[unique_classes <= max_class_val]

    # 1. 计算全域 (Global) 分类基尼
    for cls in unique_classes:
        cls_mask = valid_calc_mask & (class_array == cls)
        if np.sum(cls_mask) > 0:
            g = weighted_gini(arrays['GDP'][cls_mask], arrays['POP'][cls_mask])
            if not np.isnan(g):
                map_global[geo_plot_mask & (class_array == cls)] = g

    # 2. 计算市级嵌套 (Nested) 分类基尼
    for city_id in np.unique(city_matrix[city_matrix > 0]):
        c_mask = valid_calc_mask & (city_matrix == city_id)
        for cls in unique_classes:
            patch_mask = c_mask & (class_array == cls)
            if np.sum(patch_mask) > 0:
                g = weighted_gini(arrays['GDP'][patch_mask], arrays['POP'][patch_mask])
                if not np.isnan(g):
                    map_nested[geo_plot_mask & (city_matrix == city_id) & (class_array == cls)] = g

    return map_global, map_nested


def plot_gini_surface(data_array, title, filename):
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    gdf_city.plot(ax=ax, color='#eef2f5', edgecolor='none', zorder=0)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='none')

    # 统一使用 0.1 到 0.8 的宽容度色标展示基尼系数落差
    im = ax.imshow(data_array, cmap=cmap, vmin=0.1, vmax=0.8, extent=raster_extent,
                   interpolation='nearest', zorder=1)
    gdf_city.boundary.plot(ax=ax, color='#606060', linewidth=0.8, zorder=2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Gini Coefficient')
    ax.set_title(title, fontsize=16, pad=15)
    ax.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()


# =============================
# 3. 批量生成并渲染三大维度基尼地图
# =============================
print('>>> [2/4] 计算并渲染【所有SMOD / LCZ / GRAD】的空间基尼映射图...')

# --- SMOD ---
print('  -> 正在处理 城乡功能区 (SMOD) ...')
smod_global, smod_nested = generate_gini_maps(arrays['SMOD'])
plot_gini_surface(smod_global, '大湾区全域(九市) 城乡功能区(SMOD) 总体极化映射图', 'Map_SMOD_Global_Gini.png')
plot_gini_surface(smod_nested, '大湾区各市内部嵌套 城乡功能区(SMOD) 局部极化映射图', 'Map_SMOD_Nested_Gini.png')

# --- LCZ ---
print('  -> 正在处理 局部气候分区 (LCZ) ...')
lcz_global, lcz_nested = generate_gini_maps(arrays['LCZ'], max_class_val=17)
plot_gini_surface(lcz_global, '大湾区全域(九市) 局部气候分区(LCZ) 总体极化映射图', 'Map_LCZ_Global_Gini.png')
plot_gini_surface(lcz_nested, '大湾区各市内部嵌套 局部气候分区(LCZ) 局部极化映射图', 'Map_LCZ_Nested_Gini.png')

# --- GRAD ---
print('  -> 正在处理 建成区环境梯度 (GRAD，无截断) ...')
# 移除 max_class_val=50 限制，计算全部梯度
grad_global, grad_nested = generate_gini_maps(arrays['GRAD'])
plot_gini_surface(grad_global, '大湾区全域(九市) 建成区环境梯度(GRAD) 总体极化映射图', 'Map_GRAD_Global_Gini.png')
plot_gini_surface(grad_nested, '大湾区各市内部嵌套 建成区环境梯度(GRAD) 局部极化映射图', 'Map_GRAD_Nested_Gini.png')

# =============================
# 4. 将极化栅格地图导出为 Tiff 文件
# =============================
print('>>> [3/4] 导出基尼系数空间栅格数据 (TIFF格式) 以供GIS分析...')


def write_geotiff(out_path, array_2d, dtype, nodata_value):
    out_meta = meta.copy()
    out_meta.update({'dtype': dtype, 'nodata': nodata_value, 'compress': 'lzw'})
    write_array = array_2d.copy()
    write_array[np.isnan(write_array)] = nodata_value
    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(write_array.astype(dtype), 1)


write_geotiff(os.path.join(OUTPUT_DIR, 'Export_SMOD_Nested_Gini.tif'), smod_nested, 'float32', -9999.0)
write_geotiff(os.path.join(OUTPUT_DIR, 'Export_LCZ_Nested_Gini.tif'), lcz_nested, 'float32', -9999.0)
write_geotiff(os.path.join(OUTPUT_DIR, 'Export_GRAD_Nested_Gini.tif'), grad_nested, 'float32', -9999.0)

write_geotiff(os.path.join(OUTPUT_DIR, 'Export_SMOD_Global_Gini.tif'), smod_global, 'float32', -9999.0)
write_geotiff(os.path.join(OUTPUT_DIR, 'Export_LCZ_Global_Gini.tif'), lcz_global, 'float32', -9999.0)
write_geotiff(os.path.join(OUTPUT_DIR, 'Export_GRAD_Global_Gini.tif'), grad_global, 'float32', -9999.0)

# =============================
# 5. 提取各城市 SMOD/LCZ 平均 GDP 与热力矩阵
# =============================
print('>>> [4/4] 计算并渲染各城市不同 SMOD/LCZ 的平均 GDP 矩阵图...')
smod_types = [11, 12, 13, 21, 22, 23, 30]
lcz_types = list(range(1, 18))

matrix_smod = pd.DataFrame(index=gdf_city[col_name], columns=smod_types)
matrix_lcz = pd.DataFrame(index=gdf_city[col_name], columns=[f'LCZ_{i}' for i in lcz_types])

gdp_arr = arrays['GDP']
smod_arr = arrays['SMOD']
lcz_arr = arrays['LCZ']

for i, row in gdf_city.iterrows():
    city_id = i + 1
    cname = row[col_name]
    c_mask = (city_matrix == city_id)

    gdp_c = gdp_arr[c_mask]
    smod_c = smod_arr[c_mask]
    lcz_c = lcz_arr[c_mask]

    # 1. SMOD 平均 GDP
    for s_id in smod_types:
        valid_idx = (smod_c == s_id) & (~np.isnan(gdp_c)) & (gdp_c >= 0)
        if np.sum(valid_idx) > 0:
            matrix_smod.loc[cname, s_id] = np.mean(gdp_c[valid_idx])

    # 2. LCZ 平均 GDP
    for l_id in lcz_types:
        valid_idx = (lcz_c == l_id) & (~np.isnan(gdp_c)) & (gdp_c >= 0)
        if np.sum(valid_idx) > 0:
            matrix_lcz.loc[cname, f'LCZ_{l_id}'] = np.mean(gdp_c[valid_idx])

sns.set_theme(style='ticks', font='SimHei')

matrix_smod = matrix_smod.astype(float)
matrix_lcz = matrix_lcz.astype(float)
matrix_smod.to_csv(os.path.join(OUTPUT_DIR, 'PRD9_MeanGDP_SMOD.csv'), encoding='utf-8-sig')
matrix_lcz.to_csv(os.path.join(OUTPUT_DIR, 'PRD9_MeanGDP_LCZ.csv'), encoding='utf-8-sig')

# SMOD 平均GDP 热力图
plt.figure(figsize=(12, 8))
ax1 = sns.heatmap(matrix_smod, cmap='YlOrRd', annot=True, fmt=".2e",
                  linewidths=0.5, annot_kws={'size': 10}, mask=matrix_smod.isnull(),
                  cbar_kws={'label': 'Mean GDP per pixel (Sci Scale)'})
ax1.set_facecolor('#e0e0e0')
plt.title('珠三角九市各城乡功能区 (SMOD) 平均经济产出(GDP)异质性', fontsize=15, pad=15)
plt.xlabel('SMOD 类型代码', fontsize=12)
plt.ylabel('地级市', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_PRD9_MeanGDP_SMOD_Matrix.png'), dpi=300)
plt.close()

# LCZ 平均GDP 热力图
plt.figure(figsize=(18, 10))
ax2 = sns.heatmap(matrix_lcz, cmap='YlOrRd', annot=True, fmt=".2e",
                  linewidths=0.5, annot_kws={'size': 9}, mask=matrix_lcz.isnull(),
                  cbar_kws={'label': 'Mean GDP per pixel (Sci Scale)', 'shrink': 0.8})
ax2.set_facecolor('#e0e0e0')
plt.title('珠三角九市各局部气候分区 (LCZ) 平均经济产出(GDP)异质性', fontsize=16, pad=20)
plt.xlabel('局部气候分区 (LCZ 1-17)', fontsize=12)
plt.ylabel('地级市', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig_PRD9_MeanGDP_LCZ_Matrix.png'), dpi=300)
plt.close()

print(f'\n>>> 全域及嵌套极化制图系统执行完毕！')
print(f'>>> 共输出 6 张空间基尼系数表面图，6 个对应的 TIFF 栅格，及 2 个平均 GDP 热力矩阵。')
print(f'>>> 所有结果文件已保存至: {OUTPUT_DIR}')