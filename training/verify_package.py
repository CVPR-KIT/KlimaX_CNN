"""CPU semantic tests, syntax checks and public-bundle hygiene checks."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    forbidden = {'.ckpt', '.pt', '.pth', '.npz', '.npy', '.nc', '.pem', '.key'}
    for path in files:
        if path.suffix in forbidden or path.name.startswith('.env'):
            raise RuntimeError(f'Non-public artifact: {path.relative_to(ROOT)}')
        if path.suffix in ('.py', '.md', '.json', '.txt'):
            body = path.read_text(encoding='utf-8')
            if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', body):
                raise RuntimeError(f'IP address requires review: {path.relative_to(ROOT)}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    results = {}
    for name in ('test_model.py', 'test_loss.py'):
        proc = subprocess.run([sys.executable, str(ROOT / name)], cwd=ROOT,
                              capture_output=True, text=True)
        results[name] = {'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
        if proc.returncode:
            raise RuntimeError(f'{name}: {proc.stdout}\n{proc.stderr}')
    import torch
    report = {'passed': True, 'scope': 'public copy CPU semantics, syntax and artifact hygiene',
              'python': sys.version.split()[0], 'torch': torch.__version__,
              'gpu_smoke_on_public_copy': False,
              'fresh_pinned_environment_install_tested': False,
              'local_runner_import': 'not tested: pytorch_lightning is not installed in local CPU environment',
              'tests': results,
              'source_sha256': {p.relative_to(ROOT / 'source').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / 'source').rglob('*.py'))}}
    (ROOT / 'PACKAGE_VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'tests': len(results), 'python_files': sum(p.suffix == '.py' for p in files)}))


if __name__ == '__main__':
    main()
