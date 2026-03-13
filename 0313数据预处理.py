import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# ==========================================
# 1. 基础环境与路径配置
# ==========================================
OUTPUT_DIR = r'D:\project\城乡\data\pocesseddata\0313aligned_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f'>>> 成功建立全新输出工作空间: {OUTPUT_DIR}')

TARGET_CRS = 'EPSG:32649'
REF_SOURCE_PATH = r'D:\project\城乡\data\pocesseddata\mollweidetiff_python\裁剪以及重采样\GHS_POP.tif'
MASTER_REF_NAME = 'Master_POP_100m.tif'

PROCESS_LIST = [
    (
        r'D:\project\城乡\data\pocesseddata\mollweidetiff_python\裁剪以及重采样\GHS-SMOD1.tif',
        'Aligned_100m_SMOD.tif',
        Resampling.mode,
        'int32',
        -9999
    ),
    (
        r'D:\project\城乡\data\rawdata\lcz\lcz',
        'Aligned_100m_LCZ.tif',
        Resampling.mode,
        'int32',
        -9999
    ),
    (
        r'D:\project\城乡\data\rawdata\1210大湾区LCZ数据\2_24xj数据\PRD_Buffer_250m_nooutside_1.tif',
        'Aligned_100m_GRAD.tif',
        Resampling.nearest,
        'int32',
        -9999
    ),
    (
        r'D:\project\城乡\data\pocesseddata\gdp\gdpgba',
        'Aligned_100m_GDP.tif',
        Resampling.nearest,
        'float32',
        -3.4028235e+38
    ),
    (
        r'D:\project\城乡\data\pocesseddata\mollweidetiff_python\裁剪以及重采样\GHS_BUILT_S.tif',
        'Aligned_100m_BUILT.tif',
        Resampling.bilinear,
        'float32',
        -9999.0
    ),
    (
        r'D:\project\城乡\data\rawdata\VNL_v21_npp_2020_global_vcmslcfg_c202205302300.average_masked.dat.tif\VNL_v21_npp_2020_global_vcmslcfg_c202205302300.average_masked.dat.tif',
        'Aligned_100m_NTL.tif',
        Resampling.bilinear,
        'float32',
        -9999.0
    )
]


# ==========================================
# 2. 空间基准构建模块
# ==========================================
def build_master_reference():
    print('>>> 启动空间基准重构模块...')
    out_path = os.path.join(OUTPUT_DIR, MASTER_REF_NAME)

    with rasterio.open(REF_SOURCE_PATH) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds, resolution=100
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': TARGET_CRS,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'compress': 'lzw',
            'nodata': src.nodata
        })

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=src.nodata
            )
    print(f'>>> 基准底图生成完毕: {MASTER_REF_NAME} (维度: {height}x{width})')
    return out_path


# ==========================================
# 3. 附属栅格对齐模块
# ==========================================
def align_rasters(master_path):
    print('\n>>> 启动附属栅格空间对齐与重采样模块...')

    with rasterio.open(master_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_meta = ref.meta.copy()

    for in_path, out_name, resample_method, target_dtype, target_nodata in PROCESS_LIST:
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if not os.path.exists(in_path):
            print(f'    [跳过] 源文件不存在: {in_path}')
            continue

        print(f'    处理中 -> {out_name}')
        try:
            with rasterio.open(in_path) as src:
                kwargs = dst_meta.copy()
                kwargs.update({
                    'dtype': target_dtype,
                    'nodata': target_nodata
                })

                safe_src_nodata = src.nodata if src.nodata is not None else 0

                with rasterio.open(out_path, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resample_method,
                        src_nodata=safe_src_nodata,
                        dst_nodata=target_nodata
                    )
        except Exception as e:
            print(f'    [异常终止] {out_name}: {str(e)}')


# ==========================================
# 4. 主执行流
# ==========================================
if __name__ == '__main__':
    master_file = build_master_reference()
    align_rasters(master_file)
    print(f'\n>>> 空间预处理全流程结束。所有标准化数据已输出至: {OUTPUT_DIR}')