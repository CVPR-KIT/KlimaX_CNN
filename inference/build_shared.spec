from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs
root=Path(SPECPATH)
datas=[(str(root/'saved_model.ckpt'),'.'),(str(root/'presets.json'),'.'),
       (str(root/'stations_tc.json'),'.'),(str(root/'model_config.yaml'),'.')]
binaries=collect_dynamic_libs('netCDF4')+collect_dynamic_libs('cftime')
a=Analysis([str(root/'v1_main.py')],pathex=[str(root),str(root/'src')],binaries=binaries,datas=datas,
           hiddenimports=['motion_core_arch','preprocess_context','preprocess_v27','scipy.ndimage','netCDF4','cftime','pyproj','yaml'],
           excludes=['pytorch_lightning','lightning','torchvision','torchdata','timm','matplotlib','IPython','notebook','plotly','pandas','tensorflow'],noarchive=False)
pyz=PYZ(a.pure)
hooks=[entry for entry in a.scripts if entry[0]!='v1_main']
v1=EXE(pyz,hooks+[('v1_main',str(root/'v1_main.py'),'PYSOURCE')],[],exclude_binaries=True,
       name='KlimaX_v1',debug=False,strip=False,upx=False,console=True)
v2=EXE(pyz,hooks+[('v2_main',str(root/'v2_main.py'),'PYSOURCE')],[],exclude_binaries=True,
       name='KlimaX_v2',debug=False,strip=False,upx=False,console=True)
coll=COLLECT(v1,v2,a.binaries,a.datas,strip=False,upx=False,name='runtime')
