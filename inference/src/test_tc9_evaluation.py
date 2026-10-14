import csv
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.ascii_export import export_predictions
from src.tc9_evaluation import evaluate_tc9, read_ascii, tc9_row, write_tc9_tables


class TC9Tests(unittest.TestCase):
    def test_inclusive_hybrid_tolerance_and_zero(self):
        self.assertTrue(tc9_row('t',0.,1.)['passed'])
        self.assertIsNone(tc9_row('t',0.,0.)['absolute_error_percent'])
        self.assertTrue(tc9_row('t',4.,6.)['passed'])
        self.assertFalse(tc9_row('t',4.,6.000001)['passed'])
        self.assertFalse(tc9_row('t',0.,1.000001)['passed'])
        self.assertTrue(tc9_row('t',-1.,0.)['excluded'])
        self.assertTrue(tc9_row('t',np.nan,0.)['excluded'])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.first=datetime(2022,8,8,20,10)
        self.lat=np.linspace(37.3,37.75,100)
        self.lon=np.linspace(126.65,127.1,100)
        self.stations=[dict(station_id=sid,name='Station '+sid,lat=float(self.lat[y]),lon=float(self.lon[x]))
                       for sid,y,x in [('410',35,40),('418',42,44),('889',37,50)]]
        self.predictions=np.ones((18,100,100),dtype=np.float32)
        self.ascii=export_predictions(self.predictions,self.lat,self.lon,self.root/'output',self.first)

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self,field=None):
        anchor=self.first-timedelta(minutes=10)
        def raw(path):
            stamp=datetime.strptime(path.stem,'%Y%m%d%H%M')
            index=int((stamp-anchor).total_seconds()/600)
            result=np.full((800,800),index+10.,dtype=np.float32)
            if field is not None:field(index,result)
            return result
        with patch('src.tc9_evaluation._source_path',side_effect=lambda root,t:root/f'{t:%Y%m%d%H%M}.nc'),\
             patch('src.tc9_evaluation._read_raw',side_effect=raw),\
             patch('src.tc9_evaluation._sha256',return_value='verified'):
            return evaluate_tc9(self.root,self.first,self.ascii,self.lat,self.lon,self.stations)

    def test_actual_saved_ascii_units_complete54_and_tables(self):
        header,values=read_ascii(self.ascii/'pred_physical_t0006_EPSG3857_1km.asc')
        np.testing.assert_allclose(values[np.isfinite(values)],1.)
        result=self.evaluate()
        self.assertEqual(result['status'],'available',result.get('reason'))
        self.assertEqual(result['valid_pairs'],54)
        self.assertEqual(result['hits'],54)
        self.assertEqual(result['excluded_count'],0)
        self.assertEqual(result['mae_mm10'],0.)
        folder=write_tc9_tables(result,self.root/'tables')
        self.assertEqual((folder/'accuracy_table.csv').read_bytes()[:3],b'\xef\xbb\xbf')
        with (folder/'accuracy_table.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),54)
        with (folder/'excluded.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),0)

    def test_new_contract_truth_uses_strict_reader_and_excludes_missing(self):
        anchor=self.first-timedelta(minutes=10)
        clean=self.evaluate()
        position=(clean['mapping'][0]['native_row']+350,clean['mapping'][0]['native_col']+350)
        def strict(path,stamp):
            index=int((stamp-anchor).total_seconds()/600)
            array=np.full((800,800),index+10.,dtype=np.float32)
            if index==1: array[position]=np.nan
            return array,np.isfinite(array)
        with patch('src.tc9_evaluation._source_path',side_effect=lambda root,t:root/f'{t:%Y%m%d%H%M}.nc'),\
             patch('src.tc9_evaluation.read_raw_strict',side_effect=strict) as reader,\
             patch('src.tc9_evaluation._read_raw',side_effect=AssertionError('Legacy reader used')),\
             patch('src.tc9_evaluation._sha256',return_value='verified'):
            result=evaluate_tc9(self.root,self.first,self.ascii,self.lat,self.lon,self.stations,strict_qc=True)
        self.assertEqual(result['status'],'available',result.get('reason'))
        self.assertEqual(reader.call_count,19)
        self.assertTrue(result['rows'][0]['rows'][0]['excluded'])
        self.assertTrue(result['rows'][0]['rows'][1]['excluded'])

    def test_negative_and_missing_native_contributions_excluded(self):
        clean=self.evaluate()
        # Each station's native nearest cell contributes to its mapped mean cell.
        positions=[(m['native_row']+350,m['native_col']+350) for m in clean['mapping']]
        def invalid(index,array):
            if index==1:
                array[positions[0]]=0. # negative increment, not dry truth
                array[positions[1]]=np.nan
        result=self.evaluate(invalid)
        self.assertEqual(result['status'],'available')
        self.assertTrue(result['rows'][0]['rows'][0]['excluded'])
        self.assertTrue(result['rows'][1]['rows'][0]['excluded'])
        self.assertGreaterEqual(result['excluded_count'],2)
        self.assertEqual(result['valid_pairs']+result['excluded_count'],54)
        self.assertIsNone(result['rows'][0]['rows'][0]['truth_mm10'])

    def test_north_up_ascii_read_and_nodata_exclusion(self):
        result=self.evaluate();mapping=result['mapping'][0]
        path=self.ascii/'pred_physical_t0006_EPSG3857_1km.asc'
        text=path.read_text().splitlines();data=np.loadtxt(text[6:])
        data[mapping['asc_row'],mapping['asc_col']]=-9999.
        with path.open('w') as stream:
            stream.write('\n'.join(text[:6])+'\n');np.savetxt(stream,data,fmt='%.4f')
        result=self.evaluate()
        self.assertTrue(result['rows'][0]['rows'][0]['excluded'])
        self.assertIsNone(result['rows'][0]['rows'][0]['prediction_mm10'])
        self.assertIn('Prediction ASCII',result['rows'][0]['rows'][0]['reason'])
        self.assertLess(mapping['asc_row'],read_ascii(path)[0]['nrows'])

    def test_aggregation_and_north_flip_with_nonuniform_truth(self):
        from pyproj import Transformer
        pattern=np.indices((100,100))[0].astype(np.float32)+1
        def varied(index,array):
            array[350:450,350:450]=pattern*(index+10)
        result=self.evaluate(varied)
        header,_=read_ascii(self.ascii/'pred_physical_t0006_EPSG3857_1km.asc')
        longitude,latitude=np.meshgrid(self.lon,self.lat)
        x,y=Transformer.from_crs(4326,3857,always_xy=True).transform(longitude,latitude)
        cols=np.clip(np.floor((x-header['xllcorner'])/1000).astype(int),0,int(header['ncols'])-1)
        rows=int(header['nrows'])-1-np.clip(np.floor((y-header['yllcorner'])/1000).astype(int),0,int(header['nrows'])-1)
        for mapping,station in zip(result['mapping'],result['rows']):
            expected=pattern[(rows==mapping['asc_row'])&(cols==mapping['asc_col'])].mean()
            self.assertAlmostEqual(station['rows'][0]['truth_mm10'],float(expected),places=6)
        # Higher native latitude corresponds to lower top-origin ASC row.
        self.assertLess(result['mapping'][1]['asc_row'],result['mapping'][0]['asc_row'])

    def test_missing_future_file_never_reads_partial_truth(self):
        with patch('src.tc9_evaluation._source_path',side_effect=FileNotFoundError('missing future')),\
             patch('src.tc9_evaluation._read_raw') as reader:
            result=evaluate_tc9(self.root,self.first,self.ascii,self.lat,self.lon,self.stations)
        self.assertEqual(result['status'],'unavailable')
        self.assertNotIn('accuracy_percent',result)
        reader.assert_not_called()


if __name__=='__main__':
    unittest.main()
