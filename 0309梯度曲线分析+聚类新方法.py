import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from math import pi
import warnings
import math

warnings.filterwarnings('ignore')

# =============================
# 0. 全局配置与路径体系
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='SimHei')

BASE = r'D:\project\城乡\data\pocesseddata'

RASTERS = {
    'GDP': BASE + r'\aligned_output\Aligned_GDP_100m.tif',
    'POP': BASE + r'\aligned_output\Master_Reference_GHS_POP_100m_UTM.tif',
    'GRAD': BASE + r'\aligned_output\gradient_100m_from250m.tif'
}

VECTORS = {
    'CITY': {
        'path': BASE + r'\GBAboundary\boundarymollweide.shp',
        'field': 'name'
    },
    'COUNTY': {
        'path': BASE + r'\GBAboundary\区县\区县.shp',
        'field': 'name'
    }
}

OUTPUT_DIR = BASE + r'\0307_Final_Analysis_Result'
GAM_FIT_DIR = os.path.join(OUTPUT_DIR, '7_GAM_Fitting_Analysis')
COUNTY_GAM_DIR = os.path.join(GAM_FIT_DIR, 'County_GAM_Plots')
CLUSTER_DIR = os.path.join(OUTPUT_DIR, '8_Spatial_Clustering_Analysis')
CLUSTER_GAM_DIR = os.path.join(CLUSTER_DIR, 'Cluster_GAM_Plots')

for d in [OUTPUT_DIR, GAM_FIT_DIR, COUNTY_GAM_DIR, CLUSTER_DIR, CLUSTER_GAM_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


# =============================
# 1. 核心数学函数与GAM特征提取
# =============================
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
    return 1.0 - 2.0 * B


# ==========================================
# 核心逻辑替换：GAM拟合与寻峰算法升级
# ==========================================
def extract_gam_features(x_data, y_data):
    try:
        n_splines = max(4, min(12, len(x_data) - 2))
        gam = LinearGAM(s(0, n_splines=n_splines)).fit(x_data, y_data)

        x_fit = np.linspace(min(x_data), max(x_data), 500)
        y_fit = gam.predict(x_fit)

        G_0 = y_fit[0]
        G_min = np.min(y_fit)
        min_idx = np.argmin(y_fit)

        dy_dx = np.gradient(y_fit, x_fit)
        V_max = np.min(dy_dx)

        below_04_idx = np.where(y_fit < 0.4)[0]
        if len(below_04_idx) > 0:
            L_04 = x_fit[below_04_idx[0]]
        else:
            L_04 = np.nan

        if min_idx < len(y_fit) - 1:
            G_rebound = np.max(y_fit[min_idx:]) - y_fit[min_idx]
        else:
            G_rebound = 0.0

        # 寻峰敏感度下调：捕捉微弱反弹
        raw_peaks, _ = find_peaks(y_fit, prominence=0.01)

        valid_peaks = []
        max_x_val = np.max(x_fit)

        # 空间物理掩膜：过滤城市主极化核
        for p in raw_peaks:
            peak_x = x_fit[p]
            if peak_x > max(20.0, max_x_val * 0.25):
                valid_peaks.append(p)

        # 全局最高点残留验证剔除
        if len(valid_peaks) == 1:
            p = valid_peaks[0]
            if y_fit[p] == np.max(y_fit) and x_fit[p] < max_x_val * 0.4:
                valid_peaks = []

        peaks = np.array(valid_peaks, dtype=int)
        n_peaks = len(peaks)

        return gam, x_fit, y_fit, G_0, G_min, V_max, L_04, G_rebound, n_peaks, peaks
    except Exception as e:
        print(f"      GAM拟合警告: {e}")
        return None, None, None, np.nan, np.nan, np.nan, np.nan, np.nan, 0, []


# =============================
# 2. 空间数据读取与对齐
# =============================
print('>>> 读取栅格与矢量数据...')
arrays = {}
with rasterio.open(RASTERS['GDP']) as src:
    arrays['GDP'] = src.read(1)
    meta = src.meta
    ref_shape = src.shape

for k in ['POP', 'GRAD']:
    with rasterio.open(RASTERS[k]) as src:
        arrays[k] = src.read(1, out_shape=ref_shape, resampling=Resampling.nearest)

gdf_city = gpd.read_file(VECTORS['CITY']['path']).to_crs(meta['crs'])
rasterized_city = rasterize(
    ((geom, i) for i, geom in enumerate(gdf_city.geometry)),
    out_shape=ref_shape, transform=meta['transform'], fill=-1, dtype='int32'
)
city_map = dict(enumerate(gdf_city[VECTORS['CITY']['field']]))

gdf_county = gpd.read_file(VECTORS['COUNTY']['path']).to_crs(meta['crs'])
rasterized_county = rasterize(
    ((geom, i) for i, geom in enumerate(gdf_county.geometry)),
    out_shape=ref_shape, transform=meta['transform'], fill=-1, dtype='int32'
)
county_map = dict(enumerate(gdf_county[VECTORS['COUNTY']['field']]))

df = pd.DataFrame({
    'GDP': arrays['GDP'].ravel(),
    'POP': arrays['POP'].ravel(),
    'GRAD': arrays['GRAD'].ravel(),
    'CITY': rasterized_city.ravel(),
    'COUNTY': rasterized_county.ravel()
})
df = df[(df['GDP'] >= 0) & (df['POP'] > 0) & (df['CITY'] >= 0)]
df['CITY'] = df['CITY'].map(city_map)
df['COUNTY'] = df['COUNTY'].map(county_map)

# =============================
# 3. 市级梯度计算与GAM非参数拟合
# =============================
print('>>> 执行市级梯度基尼计算与GAM非参数拟合...')
results_params = []
fit_data_dict = {}
valid_cities = df['CITY'].dropna().unique()

for city in valid_cities:
    df_city = df[df['CITY'] == city]
    grad_stats = df_city.groupby('GRAD').apply(
        lambda x: weighted_gini(x['GDP'].values, x['POP'].values) if len(x) >= 10 else np.nan
    ).dropna()

    if len(grad_stats) < 5: continue

    x_real = grad_stats.index.values
    y_real = grad_stats.values

    dynamic_window = max(3, min(15, len(x_real) // 5))
    if dynamic_window % 2 == 0: dynamic_window += 1

    y_smooth = pd.Series(y_real).rolling(window=dynamic_window, center=True, min_periods=1).mean().values

    gam_model, x_fit, y_fit, G_0, G_min, V_max, L_04, G_rebound, n_peaks, peaks = extract_gam_features(x_real, y_smooth)

    if gam_model is not None:
        results_params.append({
            'City': city, 'Initial_Gini_G0': G_0, 'Background_Limit_Gmin': G_min,
            'Max_Decay_Rate_Vmax': V_max, 'First_Threshold_Span_L04': L_04,
            'Suburban_Rebound_Amp': G_rebound, 'Number_of_Peaks': n_peaks
        })
        fit_data_dict[city] = (x_real, y_smooth, x_fit, y_fit, L_04, peaks)

pd.DataFrame(results_params).to_excel(os.path.join(GAM_FIT_DIR, 'GAM_Topology_Parameters_City.xlsx'), index=False)

# =============================
# 4. 市级GAM拟合曲线多维可视化
# =============================
print('>>> 渲染市级GAM拟合曲线...')
n_cities = len(fit_data_dict)
cols = 4
rows = math.ceil(n_cities / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), sharex=False, sharey=True)
axes = axes.flatten()

for i, (city, (x_real, y_smooth, x_fit, y_fit, L_04, peaks)) in enumerate(fit_data_dict.items()):
    ax = axes[i]
    ax.scatter(x_real, y_smooth, color='lightgray', s=10, alpha=0.6, label='实际观测(平滑)')
    ax.plot(x_fit, y_fit, color='#2ca25f', linewidth=2.5, label='GAM拟合曲线')
    ax.axhline(y=0.4, color='#e34a33', linestyle='--', linewidth=1.2, label='0.4警戒线')

    if len(peaks) > 0:
        ax.plot(x_fit[peaks], y_fit[peaks], 'rx', markersize=8, label='次级波峰' if i == 0 else '')

    if not np.isnan(L_04):
        ax.plot(L_04, 0.4, 'ko', markersize=6)
        ax.axvline(x=L_04, color='gray', linestyle=':', linewidth=1)
        ax.text(L_04 + 1, 0.42, f'L0.4={L_04:.1f}', fontsize=10, color='black')

    ax.set_title(city, fontsize=13)
    ax.set_ylim(0.1, 0.8)
    if i == 0: ax.legend(fontsize=9, loc='upper right')

for j in range(len(fit_data_dict), len(axes)): fig.delaxes(axes[j])
plt.suptitle('大湾区城市不平等空间梯度：广义相加模型(GAM)非线性拟合', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(GAM_FIT_DIR, 'Cities_GAM_Fitting_Matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================
# 5. 区县级GAM特征提取与可视化
# =============================
print('>>> 执行区县级GAM参数提取与高维聚类...')
county_params = []
county_fit_data_dict = {}
flat_county_fit_data = {}
valid_counties = df['COUNTY'].dropna().unique()
max_grad_global = df['GRAD'].max()

for county in valid_counties:
    df_county = df[df['COUNTY'] == county]
    total_sample = len(df_county)
    parent_city = df_county['CITY'].iloc[0] if not df_county['CITY'].empty else 'Unknown'

    min_pixel_req = 3 if total_sample < 1000 else 10

    grad_stats = df_county.groupby('GRAD').apply(
        lambda x: weighted_gini(x['GDP'].values, x['POP'].values) if len(x) >= min_pixel_req else np.nan
    ).dropna()

    if len(grad_stats) < 5: continue

    x_real = grad_stats.index.values
    y_real = grad_stats.values

    dynamic_window = max(3, min(15, len(x_real) // 3))
    if dynamic_window % 2 == 0: dynamic_window += 1

    y_smooth = pd.Series(y_real).rolling(window=dynamic_window, center=True, min_periods=1).mean().values

    gam_model, x_fit, y_fit, G_0, G_min, V_max, L_04, G_rebound, n_peaks, peaks = extract_gam_features(x_real, y_smooth)

    if not np.isnan(G_0):
        if np.isnan(L_04):
            L_04 = max_grad_global if np.mean(y_smooth) > 0.4 else 0.0

        county_params.append({
            'County': county, 'Initial_G0': G_0, 'Decay_Rate_Vmax': abs(V_max) if not np.isnan(V_max) else 0,
            'Threshold_Span_L04': L_04, 'Rebound_Amp_Grebound': G_rebound, 'Number_of_Peaks': n_peaks
        })

        if parent_city not in county_fit_data_dict:
            county_fit_data_dict[parent_city] = {}
        county_fit_data_dict[parent_city][county] = (x_real, y_smooth, x_fit, y_fit, L_04, peaks)

        flat_county_fit_data[county] = {
            'city': parent_city, 'x_real': x_real, 'y_smooth': y_smooth,
            'x_fit': x_fit, 'y_fit': y_fit, 'L_04': L_04, 'peaks': peaks
        }

df_features = pd.DataFrame(county_params).set_index('County')

print('>>> 渲染区县级GAM拟合曲线(按市分组输出)...')
for city, counties in county_fit_data_dict.items():
    n_c = len(counties)
    if n_c == 0: continue
    cols = 4
    rows = math.ceil(n_c / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), sharex=False, sharey=True)
    if n_c == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i, (county, (x_real, y_smooth, x_fit, y_fit, L_04, peaks)) in enumerate(counties.items()):
        ax = axes[i]
        ax.scatter(x_real, y_smooth, color='lightgray', s=10, alpha=0.6)
        ax.plot(x_fit, y_fit, color='#2c7bb6', linewidth=2.5)
        ax.axhline(y=0.4, color='#e34a33', linestyle='--', linewidth=1.2)

        if len(peaks) > 0: ax.plot(x_fit[peaks], y_fit[peaks], 'rx', markersize=8)
        if not np.isnan(L_04):
            ax.plot(L_04, 0.4, 'ko', markersize=6)
            ax.axvline(x=L_04, color='gray', linestyle=':', linewidth=1)

        ax.set_title(county, fontsize=12)
        ax.set_ylim(0.0, 0.9)
        ax.grid(True, linestyle=':', alpha=0.5)

    for j in range(n_c, len(axes)): fig.delaxes(axes[j])
    plt.suptitle(f'{city}下辖区县空间极化GAM拟合形态', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(COUNTY_GAM_DIR, f'{city}_County_GAM.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =============================
# 6. K-Means聚类与聚类特征可视化
# =============================
if not df_features.empty and len(df_features) >= 4:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features['Cluster'] = kmeans.fit_predict(X_scaled)
    df_features.to_excel(os.path.join(CLUSTER_DIR, 'County_GAM_Clusters.xlsx'))

    print('>>> 渲染聚类特征雷达图与散点投影...')
    cluster_means = df_features.groupby('Cluster').mean()
    scaler_radar = StandardScaler()
    cluster_means_scaled = pd.DataFrame(scaler_radar.fit_transform(cluster_means), columns=cluster_means.columns,
                                        index=cluster_means.index)

    categories = ['核心极化度(G0)', '最大平抑率(Vmax)', '高不平等跨度(L0.4)', '远郊反弹幅(Grebound)',
                  '多中心波峰数(Peaks)']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=12)

    colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
    cluster_labels = ['模式A', '模式B', '模式C', '模式D']

    for i in range(n_clusters):
        values = cluster_means_scaled.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i],
                label=f'{cluster_labels[i]} (n={(df_features["Cluster"] == i).sum()})')
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('大湾区区县空间极化拓扑模式：多维参数聚类特征分布', size=15, y=1.1)
    plt.savefig(os.path.join(CLUSTER_DIR, 'Cluster_Radar_Chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df_features, x='Threshold_Span_L04', y='Rebound_Amp_Grebound', hue='Cluster', palette=colors,
                    size='Number_of_Peaks', sizes=(50, 400), alpha=0.8, edgecolor='black', ax=ax)
    for i in range(df_features.shape[0]):
        if df_features['Rebound_Amp_Grebound'].iloc[i] > df_features['Rebound_Amp_Grebound'].quantile(0.9) or \
                df_features['Threshold_Span_L04'].iloc[i] > df_features['Threshold_Span_L04'].quantile(0.9):
            ax.text(df_features['Threshold_Span_L04'].iloc[i] + 0.5, df_features['Rebound_Amp_Grebound'].iloc[i],
                    df_features.index[i], fontsize=9)
    plt.title('空间极化演化模式二维投影：警戒跨度与反弹幅度 (点大小=波峰数量)', fontsize=15)
    plt.xlabel('高不平等跨度 L0.4 (建成区梯度)', fontsize=12)
    plt.ylabel('远郊极化反弹幅度 Grebound (基尼增量)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(CLUSTER_DIR, 'Cluster_Scatter_Projection.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print('>>> 提取分组名单并渲染聚类GAM拼图矩阵...')
    for c_idx in range(n_clusters):
        counties_in_cluster = df_features[df_features['Cluster'] == c_idx].index.tolist()
        n_c = len(counties_in_cluster)
        if n_c == 0: continue

        cols = 5
        rows = math.ceil(n_c / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5), sharex=False, sharey=True)
        if n_c == 1: axes = np.array([axes])
        axes = axes.flatten()

        for i, county in enumerate(counties_in_cluster):
            ax = axes[i]
            data = flat_county_fit_data[county]
            x_real, y_smooth, x_fit, y_fit, L_04, peaks, city = data['x_real'], data['y_smooth'], data['x_fit'], data[
                'y_fit'], data['L_04'], data['peaks'], data['city']

            ax.scatter(x_real, y_smooth, color='lightgray', s=10, alpha=0.6)
            ax.plot(x_fit, y_fit, color=colors[c_idx], linewidth=2.5)
            ax.axhline(y=0.4, color='#e34a33', linestyle='--', linewidth=1.2)

            if len(peaks) > 0: ax.plot(x_fit[peaks], y_fit[peaks], 'rx', markersize=8)
            if not np.isnan(L_04):
                ax.plot(L_04, 0.4, 'ko', markersize=6)
                ax.axvline(x=L_04, color='gray', linestyle=':', linewidth=1)

            ax.set_title(f"{city}-{county}", fontsize=12)
            ax.set_ylim(0.0, 0.9)
            ax.grid(True, linestyle=':', alpha=0.5)

        for j in range(n_c, len(axes)): fig.delaxes(axes[j])
        plt.suptitle(f'空间极化演化拓扑分型 {cluster_labels[c_idx]} (共 {n_c} 个区县)', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(CLUSTER_GAM_DIR, f'Cluster_{cluster_labels[c_idx]}_Matrix.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

else:
    print('>>> 警告: 有效区县样本不足，无法执行聚类分析。')

print('>>> 所有分析模块执行完毕。')