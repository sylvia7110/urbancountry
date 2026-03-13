import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

# =============================
# 0. 全局配置与路径体系
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='SimHei')

BASE = r'D:\project\城乡\data\pocesseddata'

RASTERS = {
    'POP': BASE + r'\0313aligned_output\Master_POP_100m.tif',
    'GDP': BASE + r'\0313aligned_output\Aligned_100m_GDP.tif',  # 补充计算分子
    'SMOD': BASE + r'\0313aligned_output\Aligned_100m_SMOD.tif',
    'LCZ': BASE + r'\0313aligned_output\Aligned_100m_LCZ.tif',
    'GRAD': BASE + r'\0313aligned_output\Aligned_100m_GRAD.tif'
}

VECTORS = {
    'CITY': {
        'path': BASE + r'\GBAboundary\boundarymollweide.shp',
        'field': 'name'
    }
}

MAP_RASTER_DIR = BASE + r'\0313GINItestoutput_maps'
if not os.path.exists(MAP_RASTER_DIR):
    os.makedirs(MAP_RASTER_DIR)
with rasterio.open(r'D:\project\城乡\data\rawdata\lcz\lcz') as src: print(f"坐标系: {src.crs}\n分辨率: {src.res}\n维度: {src.shape}\n仿射变换参数:\n{src.transform}\nNoData值: {src.nodata}")
# =============================
# 1. 空间数据加载与双掩模构建
# =============================
print('>>> 启动空间数据读取与矩阵初始化...')
arrays = {}

# 以 POP 数据作为全局空间参考框架
with rasterio.open(RASTERS['POP']) as src_pop:
    meta = src_pop.meta.copy()
    ref_shape = (meta['height'], meta['width'])
    transform_val = meta['transform']
    crs_val = meta['crs']

    arrays['POP'] = src_pop.read(1).astype(np.float32)
    arrays['POP'][arrays['POP'] == src_pop.nodata] = np.nan
    arrays['POP'][arrays['POP'] < 0] = np.nan

# 读取其余栅格
for key in ['GDP', 'SMOD', 'LCZ', 'GRAD']:
    with rasterio.open(RASTERS[key]) as src:
        arrays[key] = src.read(1)

# 构建空间掩模
print('>>> 执行矢量坐标变换与双掩模解耦生成...')
gdf_city = gpd.read_file(VECTORS['CITY']['path'])

if gdf_city.crs != crs_val:
    gdf_city = gdf_city.to_crs(crs_val)

geom_list = [(geom, 1) for geom in gdf_city.geometry]
study_area_mask_int = rasterize(
    shapes=geom_list,
    out_shape=ref_shape,
    transform=transform_val,
    fill=0,
    dtype=rasterio.uint8
)

# 绘图掩模：限定于行政边界内，保留所有0值区域以维持空间形态连续性
geo_plot_mask = (study_area_mask_int == 1) & (~np.isnan(arrays['POP']))

# 计算掩模：严格提取具备人口与GDP权重的有效像元，规避底层数学逻辑失效
valid_calc_mask = geo_plot_mask & (arrays['POP'] > 0) & (~np.isnan(arrays['GDP'])) & (arrays['GDP'] >= 0)

xmin = transform_val[2]
xmax = transform_val[2] + transform_val[0] * meta['width']
ymin = transform_val[5] + transform_val[4] * meta['height']
ymax = transform_val[5]
raster_extent = [xmin, xmax, ymin, ymax]

# =============================
# 2. 空间基尼系数统计测度算法
# =============================
print('>>> 运行高维空间人口加权基尼测度模型...')

def weighted_gini(x, w):
    mask = (x >= 0) & (w > 0)
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

res_smod, res_lcz, res_grad = [], [], []

# 基于计算掩模执行数值迭代
for code in np.unique(arrays['SMOD'][valid_calc_mask]):
    mask = (arrays['SMOD'] == code) & valid_calc_mask
    res_smod.append({'SMOD': code, 'Gini': weighted_gini(arrays['GDP'][mask], arrays['POP'][mask])})

for code in np.unique(arrays['LCZ'][valid_calc_mask]):
    mask = (arrays['LCZ'] == code) & valid_calc_mask
    res_lcz.append({'LCZ_Code': code, 'Gini': weighted_gini(arrays['GDP'][mask], arrays['POP'][mask])})

for code in np.unique(arrays['GRAD'][valid_calc_mask]):
    mask = (arrays['GRAD'] == code) & valid_calc_mask
    res_grad.append({'Gradient': code, 'Gini': weighted_gini(arrays['GDP'][mask], arrays['POP'][mask])})

# =============================
# 3. 高维栅格拓扑映射与制图
# =============================
print('>>> 执行重分类映射与测度表面制图...')

def plot_raster_map(data_array, title, filename):
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='#f0f0f0')

    im = ax.imshow(data_array, cmap=cmap, vmin=0.2, vmax=0.8, extent=raster_extent)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Gini Coefficient')

    ax.set_title(title, fontsize=15, pad=15)
    ax.axis('off')

    plt.savefig(os.path.join(MAP_RASTER_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 基于制图掩模执行空间像元重分类赋值
map_smod_global = np.full(ref_shape, np.nan, dtype=np.float32)
for item in res_smod:
    mask = (arrays['SMOD'] == item['SMOD']) & geo_plot_mask
    map_smod_global[mask] = item['Gini']
plot_raster_map(map_smod_global, '大湾区城乡功能体系(SMOD)基尼系数测度表面', 'Raster_Global_SMOD.png')

map_lcz_global = np.full(ref_shape, np.nan, dtype=np.float32)
for item in res_lcz:
    mask = (arrays['LCZ'] == item['LCZ_Code']) & geo_plot_mask
    map_lcz_global[mask] = item['Gini']
plot_raster_map(map_lcz_global, '大湾区三维物理形态(LCZ)基尼系数测度表面', 'Raster_Global_LCZ.png')

map_grad_global = np.full(ref_shape, np.nan, dtype=np.float32)
for item in res_grad:
    mask = (arrays['GRAD'] == item['Gradient']) & geo_plot_mask
    map_grad_global[mask] = item['Gini']
plot_raster_map(map_grad_global, '大湾区中心边缘空间衰减梯度(GRAD)动态测度表面', 'Raster_Global_GRAD.png')

print(f'>>> 进程全部执行完毕。输出结果位于: {MAP_RASTER_DIR}')