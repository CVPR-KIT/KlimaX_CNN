from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs
root=Path(SPECPATH)
datas=[(str(root/'saved_model.ckpt'),'.'),(str(root/'presets.json'),'.'),
       (str(root/'stations_tc.json'),'.'),(str(root/'model_config.yaml'),'.'),
       (str(root/'input/data_10m_v45'),'input/data_10m_v45')]
binaries=collect_dynamic_libs('netCDF4')+collect_dynamic_libs('cftime')
a=Analysis([str(root/'v2_main.py')],pathex=[str(root),str(root/'src')],binaries=binaries,datas=datas,
           hiddenimports=['motion_core_arch','preprocess_context','preprocess_v27','scipy.ndimage','netCDF4','cftime','pyproj','yaml'],
           excludes=['pytorch_lightning','lightning','torchvision','torchdata','timm','matplotlib','IPython','notebook','plotly','pandas','tensorflow'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='KlimaX_v2',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=True)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='V2')
