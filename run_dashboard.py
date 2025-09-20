#!/usr/bin/env python3
"""
Simple Flask Dashboard Launcher
==============================

This script launches just the Flask dashboard with proper path setup.
"""

import os
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
print(f"📁 Project root: {PROJECT_ROOT}")

# Add necessary paths
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ui"))

# Change to UI directory
os.chdir(PROJECT_ROOT / "ui")

try:
    print("🌐 Starting Flask dashboard...")
    from app import app
    
    app.config['DEBUG'] = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    print("✅ Dashboard starting on http://localhost:5000")
    print("📋 Features available:")
    print("   • Face verification with explanations")
    print("   • Interactive explanation exploration")
    print("   • Accessibility features")
    print("   • Visual attention maps")
    print("")
    print("🔧 Usage:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Upload two face images")
    print("   3. Click 'Verify Identity'")
    print("   4. Explore the explanations!")
    print("")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the right directory and Flask is installed")
except Exception as e:
    print(f"❌ Error: {e}")