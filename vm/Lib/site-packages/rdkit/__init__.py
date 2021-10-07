

""""""  # start delvewheel patch
def _delvewheel_init_patch_0_0_14():
    import os
    import sys
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'rdkit_pypi.libs'))
    if sys.version_info[:2] >= (3, 8):
        if os.path.exists(os.path.join(sys.base_prefix, 'conda-meta')):
            os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']='1'
        os.add_dll_directory(libs_dir)
    else:
        from ctypes import WinDLL
        with open(os.path.join(libs_dir, '.load-order-rdkit_pypi-2021.3.5.1')) as file:
            load_order = file.read().split()
        for lib in load_order:
            WinDLL(os.path.join(libs_dir, lib))


_delvewheel_init_patch_0_0_14()
del _delvewheel_init_patch_0_0_14
# end delvewheel patch

try:
  from .rdBase import rdkitVersion as __version__
except ImportError:
  __version__ = 'Unknown'
  raise

import logging
logger = logging.getLogger("rdkit")
# if we are running in a jupyter notebook, enable the extensions
try:
  kernel_name = get_ipython().__class__.__name__

  if kernel_name == 'ZMQInteractiveShell':
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import LogWarningMsg, WrapLogs
    WrapLogs()  
    logger.info("Enabling RDKit %s jupyter extensions"%__version__)
    
except:
  pass
