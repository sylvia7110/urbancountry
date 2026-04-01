# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================
# 0. 全局配置与参数定义
# =============================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='Arial')

BASE = r'D:\project\城乡\data\pocesseddata'
ALIGN_DIR = os.path.join(BASE, r'arcgis_pocessed data(1)\arcgis_pocessed data')
OUTPUT_DIR = os.path.join(BASE, '0331cityoutputENGrevised')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RASTERS = {
    'POP': os.path.join(ALIGN_DIR, 'master_pop_100m.tif'),
    'GDP': os.path.join(ALIGN_DIR, 'aligned_100m_gdp_projectr.tif'),
    'NTL': os.path.join(ALIGN_DIR, 'extract_vnl_7_projectrast.tif'),
    'LCZ': os.path.join(ALIGN_DIR, 'aligned_100m_lcz_projectr.tif'),
    'DISTANCE': os.path.join(ALIGN_DIR, 'aligned_100m_grad_project.tif') 
}

VECTORS = {
    'CITY': {'path': os.path.join(BASE, r'GBAboundary\boundarymollweide.shp'), 'field': 'name'},
    'COUNTY': {'path': os.path.join(BASE, r'GBAboundary\区县\区县.shp'), 'field': 'name'},
    'TOWN': {'path': r'D:\project\城乡\data\区划数据\wx_GBAwithoutHKMC\GBAwithoutHM.shp', 'field': '乡'}
}

# 核心更新：直接使用梯度系数值（后续代码会自动 *0.25 转为 km 绘图）
GRADIENT_THRESHOLDS_IDX = {
    '大湾区': (4.585, 6.89, 11.66),
    '香港': (4.21, 5.41, 7.44),
    '澳门': (2.85, 4.08, 6.6),
    '广州': (3.62, 5.44, 9.25),
    '深圳': (3.81, 5.07, 7.29),
    '珠海': (2.52, 4.48, 10.05),
    '东莞': (5.22, 6.76, 9.39),
    '中山': (3.07, 4.41, 7.06),
    '佛山': (4.25, 6.69, 12.09),
    '肇庆': (2.65, 4.28, 8.19),
    '江门': (3.09, 5.57, 12.45),
    '惠州': (3.78, 5.85, 10.33)
}

CITY_EN = {
    '广州': 'Guangzhou', '深圳': 'Shenzhen', '香港': 'Hong Kong', '澳门': 'Macao',
    '珠海': 'Zhuhai', '佛山': 'Foshan', '东莞': 'Dongguan', '中山': 'Zhongshan',
    '惠州': 'Huizhou', '江门': 'Jiangmen', '肇庆': 'Zhaoqing', '大湾区': 'GBA'
}

GROUP_EN = {
    '核心城市组': 'Core Cities',
    '节点城市组': 'Node Cities',
    '边缘城市组': 'Peripheral Cities'
}

# 字符串清洗：去除"市"、"特别行政区"后缀，实现精准匹配
def get_short_city(name):
    if not isinstance(name, str): return 'Unknown'
    for short_name in GRADIENT_THRESHOLDS_IDX.keys():
        if short_name in name:
            return short_name
    return 'Unknown'

def assign_city_group(short_city):
    if short_city in ['广州', '深圳', '香港', '澳门']: return '核心城市组'
    if short_city in ['珠海', '佛山', '东莞', '中山']: return '节点城市组'
    if short_city in ['惠州', '江门', '肇庆']: return '边缘城市组'
    return 'Unknown'

# =============================
# 1. 核心数学统计函数
# =============================
def weighted_gini(values, weights):
    if len(values) < 5 or np.sum(weights) == 0: return np.nan
    order = np.argsort(values)
    v, w = values[order], weights[order]
    score = np.append([0], np.cumsum(w) / np.sum(w))
    lorenz = np.append([0], np.cumsum(v * w) / np.sum(v * w))
    return max(0.0, 1.0 - 2.0 * np.trapz(lorenz, score))

def weighted_theil(values, weights):
    valid = (values > 0) & (weights > 0)
    v, w = values[valid], weights[valid]
    if len(v) < 5 or np.sum(w) == 0: return np.nan
    y = v * w
    Y_total, P_total = np.sum(y), np.sum(w)
    if Y_total == 0: return np.nan
    s, p = y / Y_total, w / P_total
    nonzero = (s > 0) & (p > 0)
    return max(0.0, np.sum(s[nonzero] * np.log(s[nonzero] / p[nonzero])))

def theil_decomposition(df, group_col, val_col, weight_col='POP'):
    valid_mask = (df[val_col] > 0) & (df[weight_col] > 0)
    data = df[valid_mask]
    if len(data) == 0: return np.nan, np.nan, np.nan, np.nan
    Y = np.sum(data[val_col] * data[weight_col])
    P = np.sum(data[weight_col])
    if Y == 0 or P == 0: return np.nan, np.nan, np.nan, np.nan
    T_total = weighted_theil(data[val_col].values, data[weight_col].values)
    T_between, T_within = 0.0, 0.0
    for name, group in data.groupby(group_col):
        Y_k = np.sum(group[val_col] * group[weight_col])
        P_k = np.sum(group[weight_col])
        if Y_k == 0 or P_k == 0: continue
        S_k, P_share_k = Y_k / Y, P_k / P
        T_between += S_k * np.log(S_k / P_share_k)
        T_k = weighted_theil(group[val_col].values, group[weight_col].values)
        if not np.isnan(T_k): T_within += S_k * T_k
    ratio = (T_between / T_total) if T_total > 0 else 0
    return T_total, T_between, T_within, ratio

# =============================
# 2. 数据加载与预处理 (四区段更新)
# =============================
def rasterize_vector(vector_info, ref_meta):
    gdf = gpd.read_file(vector_info['path'])
    if gdf.crs != ref_meta['crs']: gdf = gdf.to_crs(ref_meta['crs'])
    field = vector_info['field']
    unique_names = gdf[field].dropna().unique()
    name_to_id = {name: idx + 1 for idx, name in enumerate(unique_names)}
    id_to_name = {idx + 1: name for idx, name in enumerate(unique_names)}
    geom_value_pairs = [(geom, name_to_id[name]) for geom, name in zip(gdf.geometry, gdf[field]) if pd.notnull(name)]
    rasterized = rasterize(geom_value_pairs, out_shape=(ref_meta['height'], ref_meta['width']),
                           transform=ref_meta['transform'], fill=0, dtype=np.uint16)
    return rasterized, id_to_name

def load_and_preprocess():
    print(">>> Step 1: Spatial Alignment & Joint Masking...")
    with rasterio.open(RASTERS['GDP']) as src:
        meta = src.meta.copy()
        gdp = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999
    with rasterio.open(RASTERS['NTL']) as src: ntl = src.read(1)
    with rasterio.open(RASTERS['POP']) as src: pop = src.read(1)
    with rasterio.open(RASTERS['LCZ']) as src: lcz = src.read(1)
    with rasterio.open(RASTERS['DISTANCE']) as src: dist = src.read(1)

    city_array, city_map = rasterize_vector(VECTORS['CITY'], meta)
    county_array, county_map = rasterize_vector(VECTORS['COUNTY'], meta)
    town_array, town_map = rasterize_vector(VECTORS['TOWN'], meta)
    
    valid_mask = ((gdp > 0) & (gdp != nodata) & (ntl > 0.5) & (ntl != nodata) &
                  (pop > 0) & (pop != nodata) & (lcz >= 1) & (lcz <= 10) &
                  (dist >= 0) & (dist <= 40) & (dist != nodata) & (city_array > 0))

    df = pd.DataFrame({
        'GDP': gdp[valid_mask], 'NTL': ntl[valid_mask], 'POP': pop[valid_mask],
        'LCZ': lcz[valid_mask], 'Gradient_Idx': dist[valid_mask], 
        'City_ID': city_array[valid_mask], 'County_ID': county_array[valid_mask], 'Town_ID': town_array[valid_mask]
    })
    
    df['Distance_km'] = df['Gradient_Idx'] * 0.25
    df['City_Raw'] = df['City_ID'].map(city_map)
    df['County'] = df['County_ID'].map(county_map)
    df['Town'] = df['Town_ID'].map(town_map)
    
    # 彻底解决匹配Bug：提取短名
    df['City_Short'] = df['City_Raw'].apply(get_short_city)
    df['City_EN'] = df['City_Short'].map(CITY_EN)
    df['City_Group'] = df['City_Short'].apply(assign_city_group)
    
    # 分配四区段宏观圈层: 1-Core, 2-Inner Urban, 3-Suburban, 4-Fringes
    df['Macro_Gradient'] = 4 # 默认为最外围
    for city_short, (th1, th2, th3) in GRADIENT_THRESHOLDS_IDX.items():
        if city_short == '大湾区': continue 
        idx = df['City_Short'] == city_short
        g_idx = df['Gradient_Idx']
        df.loc[idx & (g_idx <= th1), 'Macro_Gradient'] = 1
        df.loc[idx & (g_idx > th1) & (g_idx <= th2), 'Macro_Gradient'] = 2
        df.loc[idx & (g_idx > th2) & (g_idx <= th3), 'Macro_Gradient'] = 3

    return df

# =============================
# 3. 绘图类 1: 梯度衰减与极化游走 (含 4 区段遮罩)
# =============================
def draw_shaded_subplots(grad_df, value_cols, y_label, title_prefix, file_prefix, is_gini=False):
    for grp_cn, grp_en in GROUP_EN.items():
        cities_in_grp = [c for c in grad_df['City_Short'].unique() if assign_city_group(c) == grp_cn]
        n_cities = len(cities_in_grp)
        if n_cities == 0: continue

        cols = 2 if n_cities >= 4 else n_cities
        rows = (n_cities + 1) // 2 if n_cities >= 4 else 1
        if n_cities == 3: cols, rows = 3, 1 
        
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
        axes = np.array(axes).flatten() if n_cities > 1 else [axes]

        for i, city_short in enumerate(cities_in_grp):
            ax = axes[i]
            city_data = grad_df[grad_df['City_Short'] == city_short]
            
            # 画线
            if len(value_cols) == 2:
                sns.lineplot(x='Distance', y=value_cols[0], data=city_data, ax=ax, color='steelblue', linewidth=2.5, label='GDP Density')
                sns.lineplot(x='Distance', y=value_cols[1], data=city_data, ax=ax, color='indianred', linewidth=2.5, label='NTL Density')
            else:
                sns.lineplot(x='Distance', y=value_cols[0], data=city_data, ax=ax, color='darkorange', linewidth=2.5, label='GDP Gini')
            
            # 绘制 4 圈层背景色遮罩
            if city_short in GRADIENT_THRESHOLDS_IDX:
                th1, th2, th3 = GRADIENT_THRESHOLDS_IDX[city_short]
                km1, km2, km3 = th1 * 0.25, th2 * 0.25, th3 * 0.25
                max_dist = city_data['Distance'].max()
                
                ax.axvspan(0, km1, color='#FF9999', alpha=0.15, label='Core Area')
                ax.axvspan(km1, km2, color='#FFCC66', alpha=0.15, label='Inner Urban')
                ax.axvspan(km2, km3, color='#99CC99', alpha=0.15, label='Suburban')
                if max_dist > km3:
                    ax.axvspan(km3, max(max_dist, km3 + 0.5), color='#CCE5FF', alpha=0.15, label='Fringes')
            
            city_en = CITY_EN.get(city_short, city_short)
            ax.set_title(f'{city_en}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Distance to City Center (km)', fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            
            if is_gini: ax.set_ylim(0, 1.0)
            ax.set_xlim(left=0)
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc='upper right' if not is_gini else 'lower right')

        for j in range(i + 1, len(axes)): axes[j].set_visible(False)

        plt.suptitle(f'{grp_en} - {title_prefix}', fontsize=18, fontweight='bold', y=1.0)
        # 强制预留顶部空间给 Suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f'{file_prefix}_{grp_en.replace(" ", "_")}.png'), dpi=300)
        plt.close()

def plot_category_1_gradients(df):
    print(">>> Drawing Category 1: Gradient Decay & Gini Trajectories (English)...")
    grad_stats = []
    for (city_short, grp, dist), group in df.groupby(['City_Short', 'City_Group', 'Distance_km']):
        if dist > 15: continue 
        mean_gdp = np.average(group['GDP'], weights=group['POP'])
        mean_ntl = np.average(group['NTL'], weights=group['POP'])
        g_gdp = weighted_gini(group['GDP'].values, group['POP'].values)
        
        if mean_gdp > 0 and mean_ntl > 0:
            grad_stats.append({
                'City_Short': city_short, 'City_Group': grp, 'Distance': dist, 
                'Log_GDP_Density': np.log10(mean_gdp), 'Log_NTL_Density': np.log10(mean_ntl),
                'Gini_GDP': g_gdp
            })
    grad_df = pd.DataFrame(grad_stats).dropna()
    
    draw_shaded_subplots(grad_df, ['Log_GDP_Density', 'Log_NTL_Density'], 'Log10 (Density)', 
                         'Spatial Decay Curves', 'Fig1_Gradient_Decay', is_gini=False)
    draw_shaded_subplots(grad_df, ['Gini_GDP'], 'GDP Gini Coefficient', 
                         'Economic Polarization Trajectories', 'Fig2_Gini_Trajectory', is_gini=True)

# =============================
# 4. 绘图类 2: 物理肌理 (LCZ) 全景图
# =============================
def plot_category_2_lcz_boxplots(df):
    print(">>> Drawing Category 2: LCZ 1-10 Density Boxplots (English)...")
    plot_df = df[(df['GDP'] > 1) & (df['NTL'] > 1)].copy()
    plot_df['Log_GDP'] = np.log10(plot_df['GDP'])
    
    fig_idx = 3
    for grp_cn, grp_en in GROUP_EN.items():
        grp_df = plot_df[plot_df['City_Group'] == grp_cn]
        if grp_df.empty: continue
            
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='LCZ', y='Log_GDP', hue='City_EN', data=grp_df, linewidth=1, fliersize=0.5, palette='Set2')
        plt.title(f'Fig {fig_idx}: {grp_en} - Economic Output Heterogeneity by LCZ (1-10)', fontsize=15, fontweight='bold')
        plt.xlabel('Local Climate Zone (1-10)')
        plt.ylabel('Log10 (GDP Density)')
        plt.legend(title='City', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Fig{fig_idx}_LCZ_Boxplot_{grp_en.replace(" ", "_")}.png'), dpi=300)
        plt.close()
        fig_idx += 1

# =============================
# 5. 绘图类 3: 行政嵌套与泰尔分解
# =============================
def plot_category_3_admin_and_theil(df):
    print(">>> Drawing Category 3: Nested Admin & Theil Decomposition (English)...")
    town_gini = df.dropna(subset=['Town']).groupby(['City_Short', 'Town']).apply(lambda x: weighted_gini(x['GDP'].values, x['POP'].values)).reset_index(name='Gini')
    county_gini = df.dropna(subset=['County']).groupby(['City_Short', 'County']).apply(lambda x: weighted_gini(x['GDP'].values, x['POP'].values)).reset_index(name='Gini')
    city_gini = df.groupby('City_Short').apply(lambda x: weighted_gini(x['GDP'].values, x['POP'].values)).reset_index(name='Gini')
    
    city_order_short = city_gini.sort_values('Gini')['City_Short'].tolist()
    city_order_en = [CITY_EN.get(c, c) for c in city_order_short]
    
    town_gini['City_EN'] = town_gini['City_Short'].map(CITY_EN)
    county_gini['City_EN'] = county_gini['City_Short'].map(CITY_EN)
    city_gini['City_EN'] = city_gini['City_Short'].map(CITY_EN)
    
    plt.figure(figsize=(16, 7))
    sns.stripplot(x='City_EN', y='Gini', data=town_gini, order=city_order_en, color='lightgray', alpha=0.6, jitter=0.2, zorder=1, label='Town Level')
    sns.stripplot(x='City_EN', y='Gini', data=county_gini, order=city_order_en, color='steelblue', size=8, jitter=0, zorder=2, label='County Level')
    for i, city in enumerate(city_order_en):
        val = city_gini[city_gini['City_EN'] == city]['Gini'].values[0]
        plt.hlines(val, xmin=i-0.3, xmax=i+0.3, color='red', linewidth=3, zorder=3)
    
    plt.title('Fig 6: Nested GDP Gini Coefficients (Red Line: City Average)', fontsize=15, fontweight='bold')
    plt.xlabel('City')
    plt.xticks(rotation=45)
    plt.ylabel('GDP Gini Coefficient')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig6_Nested_Administrative_Gini.png'), dpi=300)
    plt.close()

    decomp_res = []
    dim_en_map = {'County': 'County Level', 'Macro_Gradient': 'Macro Zone', 'LCZ': 'LCZ Morphology'}
    for city_short, city_df in df.groupby('City_Short'):
        city_en = CITY_EN.get(city_short, city_short)
        for dim_col, dim_name in dim_en_map.items():
            _, b, w, _ = theil_decomposition(city_df, dim_col, 'GDP')
            decomp_res.append({'City_EN': city_en, 'Decompose_By': dim_name, 'Between-group': b, 'Within-group': w})
    
    decomp_df = pd.DataFrame(decomp_res).dropna()
    fig7, axes7 = plt.subplots(3, 1, figsize=(14, 15))
    
    for ax, dim_name in zip(axes7, dim_en_map.values()):
        dim_data = decomp_df[decomp_df['Decompose_By'] == dim_name].set_index('City_EN').reindex(city_order_en)
        dim_data[['Within-group', 'Between-group']].plot(kind='bar', stacked=True, ax=ax, color=['#99CCFF', '#FF9999'])
        ax.set_title(f'Fig 7: GDP Theil Index Decomposition (by {dim_name})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Theil Index')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig7_Stacked_Theil_Decomposition.png'), dpi=300)
    plt.close()

# =============================
# 6. 绘图类 4: 极化交叉热力矩阵 (含 4 区段映射与标题修复)
# =============================
def plot_category_4_polarization_matrices(df):
    print(">>> Drawing Category 4: Gradient-LCZ Polarization Matrices (English)...")
    # 彻底更新为 4 个区域映射
    grad_map_en = {1: 'Core Area', 2: 'Inner Urban', 3: 'Suburban', 4: 'Fringes'}
    col_order = ['Core Area', 'Inner Urban', 'Suburban', 'Fringes']
    
    matrix_records = []
    for (city_short, grp, grad, lcz), sub_df in df.groupby(['City_Short', 'City_Group', 'Macro_Gradient', 'LCZ']):
        if grad not in grad_map_en: continue
        g_gdp = weighted_gini(sub_df['GDP'].values, sub_df['POP'].values)
        city_en = CITY_EN.get(city_short, city_short)
        matrix_records.append({'City_EN': city_en, 'Group': grp, 'Macro_Gradient': grad_map_en[grad], 'LCZ': lcz, 'Gini_GDP': g_gdp})
    
    m_df = pd.DataFrame(matrix_records)
    
    # 图 8: 城市组均值矩阵
    fig8, axes8 = plt.subplots(1, 3, figsize=(20, 6)) # 稍微加宽以容纳4列
    for ax, (grp_cn, grp_en) in zip(axes8, GROUP_EN.items()):
        grp_data = m_df[m_df['Group'] == grp_cn]
        if not grp_data.empty:
            avg_mat = grp_data.groupby(['LCZ', 'Macro_Gradient'])['Gini_GDP'].mean().reset_index()
            # 强制引入 4 个 Column
            matrix = avg_mat.pivot(index='LCZ', columns='Macro_Gradient', values='Gini_GDP').reindex(index=range(1, 11), columns=col_order)
            sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f', vmin=0.2, vmax=0.9, cbar_kws={'label': 'GDP Gini'})
        ax.set_title(f'{grp_en} Average Matrix')
        ax.set_ylabel('LCZ (1-10)' if ax == axes8[0] else '')
        ax.set_xlabel('Spatial Zone')
    
    plt.suptitle('Fig 8: Average Spatial Polarization Matrices by City Group', fontsize=18, fontweight='bold', y=1.02)
    fig8.tight_layout(rect=[0, 0, 1, 0.95]) # 防裁切
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig8_Matrix_Average_Groups.png'), dpi=300)
    plt.close()

    # 图 9, 10, 11: 各单体城市矩阵
    fig_idx = 9
    for grp_cn, grp_en in GROUP_EN.items():
        grp_cities_en = m_df[m_df['Group'] == grp_cn]['City_EN'].unique()
        n_cities = len(grp_cities_en)
        if n_cities == 0: continue
            
        fig, axes = plt.subplots(1, n_cities, figsize=(6 * n_cities, 6))
        if n_cities == 1: axes = [axes] 
            
        for ax, city_en in zip(axes, grp_cities_en):
            c_data = m_df[m_df['City_EN'] == city_en]
            if not c_data.empty:
                matrix = c_data.pivot(index='LCZ', columns='Macro_Gradient', values='Gini_GDP').reindex(index=range(1, 11), columns=col_order)
                sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f', vmin=0.2, vmax=0.9, cbar_kws={'label': 'GDP Gini'})
            ax.set_title(f'{city_en}')
            ax.set_ylabel('LCZ (1-10)' if ax == axes[0] else '')
            ax.set_xlabel('Spatial Zone')
            
        plt.suptitle(f'Fig {fig_idx}: Individual Matrices - {grp_en}', fontsize=18, fontweight='bold', y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # 防裁切
        plt.savefig(os.path.join(OUTPUT_DIR, f'Fig{fig_idx}_Matrix_Individual_{grp_en.replace(" ", "_")}.png'), dpi=300)
        plt.close()
        fig_idx += 1

# =============================
# 7. 主程序执行入口
# =============================
if __name__ == '__main__':
    print(f"==========================================")
    print(f"Execution Started: Spatial Economics Phase 2 (Full English Version)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"==========================================")
    
    df_valid = load_and_preprocess()
    
    plot_category_1_gradients(df_valid)
    plot_category_2_lcz_boxplots(df_valid)
    plot_category_3_admin_and_theil(df_valid)
    plot_category_4_polarization_matrices(df_valid)
    
    print(f"==========================================")
    print(f"Execution Completed! All 11 figures have been exported to {OUTPUT_DIR}.")