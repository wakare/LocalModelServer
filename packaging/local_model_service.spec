# ruff: noqa: F821
# PyInstaller supplies Analysis, EXE and the spec globals.
from pathlib import Path

project_root = Path(SPECPATH).parent
analysis = Analysis(
    [str(project_root / "src" / "local_model_service" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[(str(project_root / "src" / "local_model_service" / "dashboard.html"),
            "local_model_service")],
    hiddenimports=[], hookspath=[], hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6", "sudachipy", "sudachidict_core", "ctranslate2", "torch",
              "transformers", "pytest", "tkinter", "kindle_anki"],
    noarchive=False, optimize=1,
)
archive = PYZ(analysis.pure)
executable = EXE(archive, analysis.scripts, [], exclude_binaries=True,
                 name="LocalModelService", console=False, debug=False, strip=False, upx=False)
bundle = COLLECT(executable, analysis.binaries, analysis.datas,
                 strip=False, upx=False, name="LocalModelService")
