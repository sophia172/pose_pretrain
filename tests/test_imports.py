#!/usr/bin/env python
"""
Test script to verify that imports work correctly for the HPE Pretraining system.
This helps ensure the code structure is portable and paths are correctly resolved.
"""
import os
import sys
from importlib import import_module
import traceback

def test_import(module_path, description):
    """Test importing a module and print result."""
    print(f"Testing import: {module_path} ({description})")
    try:
        module = import_module(module_path)
        print(f"✅ Successfully imported {module_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_path}: {str(e)}")
        traceback.print_exc()
        return False

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 70 + "\n")

def main():
    """Run the import tests."""
    print("\n🧪 TESTING IMPORT STRUCTURE 🧪\n")
    
    # Track success count
    success_count = 0
    total_tests = 0
    
    print_separator()
    print("ENVIRONMENT INFO:")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print_separator()
    
    # Test absolute imports (from outside src directory)
    print("TESTING ABSOLUTE IMPORTS (from project root):")
    
    tests = [
        ("src.data.dataloader", "DataLoader module"),
        ("src.models.pretrain_model", "Transformer model"),
        ("src.models.pretrain_vit_model", "Vision Transformer model"),
        ("src.utils.device", "Device utilities"),
        ("src.trainers.trainer", "Trainer module"),
        ("src.models.components.vision_transformer", "Vision Transformer components"),
    ]
    
    for module, desc in tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1
            
    print_separator()
    
    # Add src to path for relative imports
    if os.path.basename(os.getcwd()) != "src":
        os.chdir("src")
        
    # Make sure src is in sys.path
    if "." not in sys.path:
        sys.path.insert(0, ".")
    
    print("TESTING RELATIVE IMPORTS (from inside src directory):")
    print(f"Working directory changed to: {os.getcwd()}")
    
    tests = [
        ("data.dataloader", "DataLoader module"),
        ("models.pretrain_model", "Transformer model"),
        ("models.pretrain_vit_model", "Vision Transformer model"),
        ("utils.device", "Device utilities"),
        ("trainers.trainer", "Trainer module"),
        ("models.components.vision_transformer", "Vision Transformer components"),
    ]
    
    for module, desc in tests:
        total_tests += 1
        if test_import(module, desc):
            success_count += 1
    
    print_separator()
    print(f"SUMMARY: {success_count}/{total_tests} import tests passed")
    
    if success_count == total_tests:
        print("\n✅ ALL TESTS PASSED! The project structure is correct.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Please check the import structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 