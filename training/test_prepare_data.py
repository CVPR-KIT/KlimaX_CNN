"""Data-free regression checks for raw-contract auditing and unknown coverage."""
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_data as prep
import nc_input
from raw_contract import QC_VERSION


class Variable:
    shape=(2049,2049)
    dtype=np.dtype('int16')
    data_scale=10.
    unit='mm/h'
    name='data'
    transform='value = data / data_scale'

    def __init__(self, values=None):
        self.values=np.full((800,800),100,np.float32) if values is None else values

    def __getitem__(self, key):
        return self.values

    def ncattrs(self):
        return list(vars(self))+['data_scale','unit','name','transform']

    def getncattr(self,key):
        return 'day rain' if key=='name' else getattr(self,key)


class Dataset:
    product_type='SFC_GRID_rn_day_'
    map_pro='Lambert Conformal Conic Projection'

    def __init__(self, timestamp, variable=None):
        self.time=[timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,0]
        self.variables={'data':variable or Variable()}
        for k,v in prep.GEOMETRY.items():setattr(self,k,v)

    def __getitem__(self,key):return self.variables[key]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def set_auto_maskandscale(self,*args):pass


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.timestamp=datetime(2022,8,2,0,10)

    def test_header_rejects_converted_units_time_scale_and_geometry(self):
        prep.validate_header(Dataset(self.timestamp),self.timestamp)
        mutations=[('product_type','SFC_GRID_rn_10m_'),('time',[2022,8,2,0,20,0]),
                   ('grid_size',1.),('map_pro','unknown')]
        for key,value in mutations:
            ds=Dataset(self.timestamp);setattr(ds,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):prep.validate_header(ds,self.timestamp)
        for key,value in [('unit','mm/10min'),('data_scale',1.),('scale_factor',.1),('shape',(100,100))]:
            variable=Variable();setattr(variable,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):prep.validate_header(Dataset(self.timestamp,variable),self.timestamp)

    def test_missing_disjoint_and_existing_out_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/'raw';raw.mkdir();out=Path(temp)/'dataset'
            with self.assertRaises(ValueError):prep.prepare(raw,out,'20220802','20220802')
            self.assertFalse(out.exists())
            with self.assertRaises(FileNotFoundError):prep.prepare(raw,out,'20220802','20220809')
            self.assertFalse(out.exists())
            out.mkdir()
            with self.assertRaises(FileExistsError):prep.prepare(raw,out,'20220802','20220809')

    def test_masks_do_not_relabel_unknown_intervals_as_dry(self):
        a=np.full((800,800),100,np.float32);a[400,400]=-9999
        b=np.full((800,800),110,np.float32)
        sources=[(self.timestamp,Path('a.nc')),(self.timestamp+timedelta(minutes=10),Path('b.nc'))]
        datasets=[Dataset(t,Variable(v)) for (t,_),v in zip(sources,(a,b))]
        with patch('netCDF4.Dataset',side_effect=datasets),patch.object(prep,'sha256',return_value='test-only'):
            rain,context,valid,native,provenance=prep.build_day('20220802',sources)
        self.assertFalse(native[1,50,50]);self.assertFalse(native[2,50,50])
        self.assertEqual(rain[1,50,50],0);self.assertEqual(rain[2,50,50],0)
        self.assertLess(valid[1,100,100],255);self.assertLess(valid[2,100,100],255)
        self.assertTrue(np.all(valid[0]==0));self.assertFalse(native[0].any())
        self.assertEqual(provenance[0]['invalid_native_count'],1)

    def test_valid_data_matches_actual_inference_processor(self):
        values=[np.full((800,800),n,np.float32) for n in (100,110,105,120)]
        sources=[(self.timestamp+timedelta(minutes=10*i),Path(str(i)+'.nc')) for i in range(4)]
        datasets=[Dataset(t,Variable(v)) for (t,_),v in zip(sources,values)]
        with patch('netCDF4.Dataset',side_effect=datasets),patch.object(prep,'sha256',return_value='test-only'):
            rain,context,valid,native,_=prep.build_day('20220802',sources)
        processor=prep.ResetSafeMethod4C()
        for i,((timestamp,_),raw) in enumerate(zip(sources,values),1):
            expected=prep._convert(processor.update(raw/10,timestamp))
            for actual,wanted in zip((rain[i],context[i],valid[i]),expected):np.testing.assert_array_equal(actual,wanted)
            self.assertTrue(native[i].all())

    def test_strict_training_inference_gap_and_positive_missing_value_equal(self):
        values=[np.full((800,800),100+10*i,np.float32) for i in range(6)]
        values[2][400,400]=-9999
        values[3][402,402]=5000  # positive missing sentinel below the legacy9999 cutoff
        sources=[(self.timestamp+timedelta(minutes=10*i),Path((self.timestamp+timedelta(minutes=10*i)).strftime('%Y%m%d%H%M.nc'))) for i in range(6)]
        datasets={path.name:Dataset(timestamp,Variable(value)) for (timestamp,path),value in zip(sources,values)}
        for ds in datasets.values():ds['data'].missing_value=5000
        def dataset(path):return datasets[Path(path).name]
        with tempfile.TemporaryDirectory() as temp,patch('netCDF4.Dataset',side_effect=dataset),patch.object(prep,'sha256',return_value='test-only'):
            rain,context,valid,native,_=prep.build_day('20220802',sources)
            def source(root,timestamp):return Path(timestamp.strftime('%Y%m%d%H%M.nc'))
            with patch.object(nc_input,'_source_path',side_effect=source),patch.object(nc_input,'_sha256',return_value='test-only'),patch.object(nc_input,'_save_nc'):
                loaded=nc_input.prepare_input(Path(temp),self.timestamp+timedelta(minutes=60),Path(temp)/'nc',Path(temp)/'npz',strict_qc=True)
            np.testing.assert_array_equal(loaded['fine'][:,0],rain[1:7])
            np.testing.assert_array_equal(loaded['context'][:,0],context[1:7].astype(np.float32))
            np.testing.assert_array_equal(loaded['valid'][:,0],valid[1:7].astype(np.float32)/255)
            self.assertEqual(loaded['provenance']['version'],QC_VERSION)
            self.assertFalse(native[3,50,50]);self.assertFalse(native[4,50,50])
            self.assertFalse(native[4,52,52]);self.assertFalse(native[5,52,52])
            self.assertTrue(native[6,50,50]);self.assertTrue(native[6,52,52])

    def test_windows_and_conservative_containing_cell_mask(self):
        rain=np.ones((144,100,100),np.float32);context=np.ones((144,200,200),np.float16)
        valid=np.full((144,200,200),255,np.uint8);native=np.ones((144,100,100),bool)
        refs,qc=prep.audit_windows(rain,context,valid,native,min_fine=1.,min_context=1.)
        self.assertEqual(refs,list(range(6,126)))
        valid[20,100,100]=239
        refs,qc=prep.audit_windows(rain,context,valid,native,min_fine=1.,min_context=.8)
        self.assertTrue(all(not(ref-5<=20<=ref+18) for ref in refs))
        self.assertLess(qc['fine_known_fraction'][20],1.)
        self.assertTrue(qc['excluded_windows'])

    def test_coordinate_inverse_roundtrip_and_north_up(self):
        lat,lon=prep.build_grid(prep.GEOMETRY)
        self.assertEqual(lat.shape,(100,100));self.assertTrue((np.diff(lat,axis=0)>0).all())
        d=np.pi/180;n=np.log(np.cos(30*d)/np.cos(60*d))/np.log(np.tan(np.pi/4+60*d/2)/np.tan(np.pi/4+30*d/2))
        factor=np.tan(np.pi/4+30*d/2)**n*np.cos(30*d)/n;r=6371.00877/.5
        rho=r*factor/np.tan(np.pi/4+lat.astype(float)*d/2)**n
        rho0=r*factor/np.tan(np.pi/4+38*d/2)**n
        theta=(lon.astype(float)-126)*d*n
        xx=rho*np.sin(theta)+880;yy=rho0-rho*np.cos(theta)+1540
        np.testing.assert_allclose(xx,np.broadcast_to(np.arange(998,1098),(100,100)),atol=.002)
        np.testing.assert_allclose(yy,np.broadcast_to(np.arange(1396,1496)[:,None],(100,100)),atol=.002)

    def test_supplied_grid_preserved_and_bad_region_rejected(self):
        # Stub raw reads only; exercise actual output formats, audit manifest,
        # supplied coordinates, source binding and overwrite protection.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);grid=root/'grid';grid.mkdir()
            lat,lon=prep.build_grid(prep.GEOMETRY)
            lat=lat.astype(np.float64)+.0001
            np.save(grid/'lat.npy',lat);np.save(grid/'lon.npy',lon)
            rain=np.ones((144,100,100),np.float32)
            ctx=np.ones((144,200,200),np.float16)
            valid=np.full((144,200,200),255,np.uint8)
            native=np.ones((144,100,100),bool)
            rain[0]=0;ctx[0]=0;valid[0]=0;native[0]=False
            def source(raw,timestamp):return raw/timestamp.strftime('%Y%m%d%H%M.nc')
            def dataset(path):return Dataset(datetime.strptime(path.stem,'%Y%m%d%H%M'))
            with patch.object(prep,'_source_path',side_effect=source),patch('netCDF4.Dataset',side_effect=dataset),patch.object(prep,'build_day',return_value=(rain,ctx,valid,native,[])):
                result=prep.prepare(root/'raw',root/'out','20220802','20220809',grid_source_dir=grid)
                self.assertTrue(result['audit_pass'])
                self.assertEqual(result['counts'],dict(train_windows=120,val_windows=120))
                np.testing.assert_array_equal(np.load(root/'out/grid/lat.npy'),lat)
                self.assertEqual(prep.sha256(root/'out/grid/lat.npy'),prep.sha256(grid/'lat.npy'))
                self.assertEqual(result['days'][0]['sha256'],prep.sha256(root/'out/train/20220802.npz'))
                np.save(grid/'lat.npy',lat+.1)
                with self.assertRaises(ValueError):
                    prep.prepare(root/'raw',root/'bad','20220802','20220809',grid_source_dir=grid)
                self.assertFalse((root/'bad').exists())


if __name__=='__main__':
    unittest.main()
