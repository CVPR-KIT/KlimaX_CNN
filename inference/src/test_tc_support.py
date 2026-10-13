from datetime import datetime,timedelta,timezone
import tempfile
from pathlib import Path
import unittest

from src.tc_support import record_tc8


class TimingTests(unittest.TestCase):
    def test_v1_repeat_average_and_v2_history_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=datetime(2026,10,13,tzinfo=timezone(timedelta(hours=9)))
            for i,seconds in enumerate((8.,10.,12.),1):
                folder=root/f'v1_case_run{i}';folder.mkdir()
                result=record_tc8(folder,{'mode':'v1','model':'same'},seconds,first,first+timedelta(seconds=seconds))
            self.assertTrue(result['complete'])
            self.assertEqual(result['used_count'],3)
            self.assertEqual(result['mean_seconds'],10.)
            other=root/'v2_case';other.mkdir()
            result=record_tc8(other,{'mode':'v2','model':'same'},9.,first,first+timedelta(seconds=9))
            self.assertFalse(result['complete'])
            self.assertEqual(result['used_count'],1)

    def test_only_three_matching_conditions_receive_a_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=datetime(2026,10,13,tzinfo=timezone(timedelta(hours=9)))
            results=[]
            for i,seconds in enumerate([10.,20.,30.],1):
                out=root/f'v2_case_run{i}';out.mkdir()
                results.append(record_tc8(out,{'checkpoint':'a','grid':'a','device':'cpu'},seconds,first,first+timedelta(seconds=i)))
            self.assertIsNone(results[0]['passed']);self.assertIsNone(results[1]['passed'])
            self.assertTrue(results[2]['passed']);self.assertEqual(results[2]['mean_seconds'],20.)
            out=root/'v2_other';out.mkdir()
            other=record_tc8(out,{'checkpoint':'a','grid':'b','device':'cpu'},1000.,first,first+timedelta(seconds=4))
            self.assertFalse(other['complete']);self.assertIsNone(other['passed'])

    def test_slow_mean_fails_and_invalid_records_are_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);now=datetime.now(timezone.utc)
            malformed=root/'v2_broken';malformed.mkdir();(malformed/'TC8.json').write_text('broken')
            for i,seconds in enumerate([300.,300.,301.],1):
                out=root/f'v2_case_run{i}';out.mkdir()
                result=record_tc8(out,{'case':1},seconds,now,now+timedelta(seconds=i))
            self.assertEqual(result['eligible_count'],3);self.assertFalse(result['passed'])


if __name__=='__main__': unittest.main()
