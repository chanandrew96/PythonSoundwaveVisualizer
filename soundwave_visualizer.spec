# PyInstaller spec for Soundwave Visualizer GUI (main.py)
# Run: pyinstaller soundwave_visualizer.spec

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'soundwave_visualizer',
        'soundwave_visualizer.visualizer',
        'soundwave_visualizer.defaults',
        'soundwave_visualizer.cli',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'numpy',
        'scipy',
        'scipy.fft',
        'scipy.ndimage',
        'scipy.io',
        'scipy.io.wavfile',
        'librosa',
        'PIL',
        'PIL.Image',
        'PIL.ImageFilter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SoundwaveVisualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI only, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
