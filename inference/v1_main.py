"""Prediction-only V1: same model/correction/export as V2; never evaluate future NC."""
from v2_main import main

if __name__=='__main__':
    main(mode='v1')
