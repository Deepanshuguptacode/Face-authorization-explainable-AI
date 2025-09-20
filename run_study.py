#!/usr/bin/env python3
"""
Simple Study Server Launcher
============================

This script launches just the study server with proper path setup.
"""

import os
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
print(f"📁 Project root: {PROJECT_ROOT}")

# Add necessary paths
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "human_studies"))

# Change to human_studies directory
os.chdir(PROJECT_ROOT / "human_studies")

try:
    print("📊 Starting study server...")
    from study_server import app
    
    app.config['DEBUG'] = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    print("✅ Study server starting on http://localhost:5001")
    print("📋 Features available:")
    print("   • Participant registration")
    print("   • Informed consent process")
    print("   • Demographics questionnaire")
    print("   • Explanation evaluation tasks")
    print("   • Data collection and export")
    print("")
    print("🔧 Usage:")
    print("   1. Open http://localhost:5001 in your browser")
    print("   2. Go through the consent process")
    print("   3. Fill out demographics")
    print("   4. Complete evaluation tasks")
    print("   5. Submit your feedback!")
    print("")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the right directory and Flask is installed")
except Exception as e:
    print(f"❌ Error: {e}")