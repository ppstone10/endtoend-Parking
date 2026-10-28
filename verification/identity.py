"""候选版本的身份：旧冻结依赖逐文件保持，新所有者单独登记。"""
from pathlib import Path
from scripts.verify_closed_loop_components import verification_code_identity
from scripts.diagnose_closed_loop_failures import _sha256

ROOT = Path(__file__).resolve().parents[1]


def execution_identity(extra_files):
    paths = [ROOT / 'verification/__init__.py', Path(__file__), *(ROOT / p for p in extra_files)]
    return {**verification_code_identity(), **{str(p.resolve()): _sha256(p) for p in paths}}


def verify_files(files):
    for path, digest in files.items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f'冻结文件漂移：{path}')
