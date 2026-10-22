try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata

try:
    __version__ = metadata.version("sc-complete-metrics")
except metadata.PackageNotFoundError:  # installed under the old name or run from source
    try:
        __version__ = metadata.version("scib")
    except metadata.PackageNotFoundError:
        from pathlib import Path

        _version_file = Path(__file__).resolve().parent.parent / "VERSION.txt"
        __version__ = (
            _version_file.read_text().strip() if _version_file.exists() else "unknown"
        )

from . import integration, metrics, preprocessing, utils
from ._package_tools import rename_func
from .metrics import clustering

alias_func_map = {
    "runScanorama": integration.scanorama,
    "runTrVae": integration.trvae,
    "runTrVaep": integration.trvaep,
    "runScGen": integration.scgen,
    "runScvi": integration.scvi,
    "runScanvi": integration.scanvi,
    "runMNN": integration.mnn,
    "runBBKNN": integration.bbknn,
    "runSaucie": integration.saucie,
    "runCombat": integration.combat,
    "runDESC": integration.desc,
    "readConos": preprocessing.read_conos,
    "readSeurat": preprocessing.read_seurat,
    "saveSeurat": preprocessing.save_seurat,
}

for alias, func in alias_func_map.items():
    rename_func(func, alias)

pp = preprocessing
ig = integration
me = metrics
cl = clustering