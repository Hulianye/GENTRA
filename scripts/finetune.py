#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if (
    os.environ.get('PYTHONHASHSEED') != '0'
    or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gentra.pipelines.cli_finetune import main
if __name__ == '__main__':
    main()
