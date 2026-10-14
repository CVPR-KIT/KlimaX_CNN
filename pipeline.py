"""Repository-root entry point for data preparation, training and inference."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {'prepare': 'training/prepare_data.py', 'train': 'training/train_pipeline.py',
            'predict': 'inference/v1_main.py', 'evaluate': 'inference/v2_main.py'}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print('Usage: python pipeline.py {prepare|train|predict|evaluate} [options]')
        print('Run a stage with --help for its options. Raw data is supplied locally.')
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ('--help','-h') else 2
    return subprocess.call([sys.executable,str(ROOT/COMMANDS[sys.argv[1]]),*sys.argv[2:]])


if __name__ == '__main__': raise SystemExit(main())
