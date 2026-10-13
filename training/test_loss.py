"""Convenience entrypoint for the frozen-source V45 loss regression suite."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'source' / 'experiments' / 'scripts' / 'test_loss.py'), run_name='__main__')
