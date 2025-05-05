"""
Path setup utility for the HPE Pretraining System.

This module ensures that both the src directory and project root
are in the Python path, making imports work consistently regardless
of where scripts are run from.

Usage:
    import path_setup  # At the top of any script
"""
import os
import sys
import inspect

# Get the caller's file path
caller_file = inspect.stack()[1].filename
caller_dir = os.path.dirname(os.path.abspath(caller_file))

# Detect if the caller is in the src directory or a subdirectory
in_src = 'src' in caller_dir.split(os.path.sep)

# Find the src directory and project root
if in_src:
    # If called from inside src or a subdirectory
    src_parts = caller_dir.split(os.path.sep + 'src' + os.path.sep)
    if len(src_parts) > 1:
        # Called from a subdirectory of src
        src_dir = os.path.join(src_parts[0], 'src')
    else:
        # Called directly from src
        src_dir = caller_dir
    project_root = os.path.dirname(src_dir)
else:
    # If called from outside src (e.g., project root)
    project_root = os.path.abspath(os.path.dirname(caller_file))
    src_dir = os.path.join(project_root, 'src')

# Ensure src directory is in the Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Ensure project root is in the Python path for absolute imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# For debugging (remove in production or set to False)
DEBUG = False
if DEBUG:
    print(f"path_setup: Added to sys.path:")
    print(f"  - Project root: {project_root}")
    print(f"  - Src directory: {src_dir}")
    print(f"  - Current sys.path: {sys.path}") 