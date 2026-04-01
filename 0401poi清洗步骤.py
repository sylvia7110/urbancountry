# -*- coding: utf-8 -*-
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局路径配置
# ==========================================
# POI 原始解压后存放的根目录 (根据您提供的完整目录结构)
POI_BASE_DIR = r"D:\BaiduNetdiskDownload\2020poiGBA"

# 参考栅格（模板）路径：强制对齐该影像的空间范围、分辨率与投影
REF_RASTER_PATH = r"D:\project\城乡\data\pocesseddata\arcgis_pocessed data(1)\arcgis_pocessed data\master_pop_100m.tif"

# 栅格输出目录 (直接输出至您的分析文件夹)
OUT_DIR = r"D:\project\城乡\data\pocesseddata\arcgis_pocessed data(1)\arcgis_pocessed data"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 2. 类别定义 (已剔除 汽车销售、商务住宅及非实体定位设施)
# ==========================================
CATEGORY_MAPPING = {
    'Productive': ['公司企业', '金融保险服务'],
    'Living': ['餐饮服务', '购物服务', '生活服务', '住宿服务', '汽车服务', '汽车维修', '摩托车服务'],
    'Public': ['科教文化服务', '医疗保健服务', '公共设施', '政府机构及社会团体', '体育休闲服务', '风景名胜', '交通设施服务']
}

def get_category(filename):
    for cat, kws in CATEGORY_MAPPING.items():
        if any(kw in filename for kw in kws):
            return cat
    return None

def extract_and_merge_poi(base_dir):
    """遍历子文件夹并提取分类POI点位"""
    print(">>> 步骤 1: 正在扫描并读取 11 个城市的 POI 矢量数据...")
    gdf_list = []
    path_obj = Path(base_dir)
    
    for shp_path in path_obj.rglob('*.shp'):
        filename = shp_path.name
        cat = get_category(filename)
        
        if cat:
            try:
                # 仅读取几何属性以防止内存溢出
                gdf = gpd.read_file(shp_path, ignore_geometry=False)[['geometry']].copy()
                gdf['poi_type'] = cat
                
                # 设定初始坐标系
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                else:
                    gdf.to_crs(epsg=4326, inplace=True)
                    
                gdf_list.append(gdf)
            except Exception as e:
                print(f"  [警告] 读取 {filename} 失败: {e}")

    print(">>> 步骤 2: 合并全湾区 POI 数据...")
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs="EPSG:4326")
    return merged_gdf

def rasterize_aligned_pois(poi_gdf, ref_path, out_dir):
    """严格参照模板栅格生成对齐的 POI 密度矩阵"""
    print(">>> 步骤 3: 读取参考栅格元数据 (投影、外包框、仿射变换矩阵)...")
    with rasterio.open(ref_path) as src:
        ref_meta = src.meta.copy()
        ref_shape = (src.height, src.width)
        ref_transform = src.transform
        ref_crs = src.crs

    # 强制投影转换至与基准栅格完全一致
    if poi_gdf.crs != ref_crs:
        print(f">>> 步骤 4: 重投影 POI 点位至 {ref_crs}...")
        poi_gdf = poi_gdf.to_crs(ref_crs)

    # 更新输出 TIF 元数据
    ref_meta.update({
        'dtype': 'float32',
        'nodata': 0.0, # POI密度没有数据的地方实际上是0
        'compress': 'lzw'
    })

    categories = ['Total', 'Productive', 'Living', 'Public']
    
    print(">>> 步骤 5: 开始执行空间像元计数 (Rasterize) ...")
    for cat in categories:
        if cat == 'Total':
            target_points = poi_gdf
        else:
            target_points = poi_gdf[poi_gdf['poi_type'] == cat]
            
        if target_points.empty:
            continue

        # 为每个点赋予计数值 1.0
        geom_value_pairs = [(geom, 1.0) for geom in target_points.geometry]

        # Rasterize：多个点落入同一 100m 栅格时，使用 MergeAlg.add 进行加和操作
        rasterized_array = rasterize(
            shapes=geom_value_pairs,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=0.0,
            default_value=1.0,
            dtype=np.float32,
            merge_alg=MergeAlg.add
        )

        out_tif = os.path.join(out_dir, f'aligned_100m_poi_{cat.lower()}.tif')
        with rasterio.open(out_tif, 'w', **ref_meta) as dest:
            dest.write(rasterized_array, 1)
        
        print(f"  - 导出成功: {out_tif} (最大单像元聚集数: {rasterized_array.max()})")

if __name__ == "__main__":
    # 执行流水线
    gba_pois = extract_and_merge_poi(POI_BASE_DIR)
    rasterize_aligned_pois(gba_pois, REF_RASTER_PATH, OUT_DIR)
    print(">>> 全流程完成！对齐栅格已准备就绪。")