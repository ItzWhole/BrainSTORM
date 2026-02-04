# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['bin/storm_gui.py'],
    pathex=[current_dir],
    binaries=[],
    datas=[
        ('storm_core', 'storm_core'),
        ('bin/detectionalgo.py', 'bin'),
        ('requirements.txt', '.'),
        ('README.md', '.'),
    ],
    hiddenimports=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'scipy',
        'skimage',
        'tifffile',
        'pandas',
        'PIL',
        'tkinter',
        'queue',
        'threading',
        'pathlib',
        'storm_core.data_processing',
        'storm_core.neural_network', 
        'storm_core.evaluation',
        'detectionalgo',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='STORM_Analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)