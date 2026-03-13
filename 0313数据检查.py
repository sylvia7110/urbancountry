import os
import rasterio
import geopandas as gpd
import pandas as pd

# =============================
# 0. 全局配置与路径体系
# =============================
BASE = r'D:\project\城乡\data\pocesseddata'

RASTERS = {
    'GDP': BASE + r'\0313aligned_output\Aligned_100m_GDP.tif',
    'POP': BASE + r'\0313aligned_output\Master_POP_100m.tif',
    'SMOD': BASE + r'\0313aligned_output\Aligned_100m_SMOD.tif',
    'LCZ': BASE + r'\0313aligned_output\Aligned_100m_LCZ.tif',
    'GRAD': BASE + r'\0313aligned_output\Aligned_100m_GRAD.tif'
}

VECTORS = {
    'CITY': {
        'path': BASE + r'\GBAboundary\boundarymollweide.shp',
        'field': 'name'
    },
    'COUNTY': {
        'path': BASE + r'\GBAboundary\区县\区县.shp',
        'field': 'name'
    },
    'TOWN': {
        'path': r'D:\project\城乡\data\区划数据\wx_GBAwithoutHKMC\GBAwithoutHM.shp',
        'field': '乡'
    }
}

# =============================
# 1. 栅格数据元数据提取与检验
# =============================
def check_rasters():
    raster_info = []
    for name, path in RASTERS.items():
        if not os.path.exists(path):
            raster_info.append({'数据层': name, '状态': '文件丢失', '路径': path})
            continue
        try:
            with rasterio.open(path) as src:
                raster_info.append({
                    '数据层': name,
                    '状态': '正常读取',
                    'CRS': str(src.crs),
                    '矩阵维度 (行, 列)': f'{src.height}, {src.width}',
                    '空间分辨率': f'{src.res[0]}, {src.res[1]}',
                    '数据类型': src.dtypes[0],
                    'NoData标识': src.nodata
                })
        except Exception as e:
            raster_info.append({'数据层': name, '状态': f'读取异常: {str(e)}'})
    return pd.DataFrame(raster_info)

# =============================
# 2. 矢量数据元数据提取与检验
# =============================
def check_vectors():
    vector_info = []
    for name, info in VECTORS.items():
        path = info['path']
        if not os.path.exists(path):
            vector_info.append({'数据层': name, '状态': '文件丢失', '路径': path})
            continue
        try:
            gdf = gpd.read_file(path)
            columns_sample = ', '.join(gdf.columns.tolist()[:5])
            vector_info.append({
                '数据层': name,
                '状态': '正常读取',
                'CRS': str(gdf.crs),
                '几何类型': str(gdf.geom_type.unique()),
                '要素数量': len(gdf),
                '属性表字段样本': columns_sample
            })
        except Exception as e:
            vector_info.append({'数据层': name, '状态': f'读取异常: {str(e)}'})
    return pd.DataFrame(vector_info)

# =============================
# 3. 输出诊断报告
# =============================
if __name__ == '__main__':
    print('>>> 启动空间数据一致性诊断程序...')

    df_rasters = check_rasters()
    print('\n[栅格数据源检验报告]')
    print(df_rasters.to_markdown(index=False))

    df_vectors = check_vectors()
    print('\n[矢量数据源检验报告]')
    print(df_vectors.to_markdown(index=False))

    print('\n>>> 诊断完成。请重点比对各栅格层的矩阵维度是否完全一致。')