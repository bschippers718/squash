#!/usr/bin/env python3
"""
Setup script for the YOLO + Gemini analysis pipeline
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def setup_gemini_api():
    """Help user set up Gemini API key"""
    print("\n" + "="*50)
    print("GEMINI API SETUP")
    print("="*50)
    print("To use Gemini 2.5 for analysis, you need an API key.")
    print("\nSteps:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Set it as an environment variable:")
    print("   export GEMINI_API_KEY='your_api_key_here'")
    print("\nOr you can set it when running the analysis script.")
    print("="*50)

def main():
    """Main setup function"""
    print("YOLO + Gemini Analysis Pipeline Setup")
    print("="*40)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Setup Gemini API
    setup_gemini_api()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("Next steps:")
    print("1. Set your Gemini API key (see instructions above)")
    print("2. Run detection: python3 video/detect_video.py")
    print("3. Run analysis: python3 video/gemini_analysis.py")
    print("\nThe pipeline will:")
    print("- Detect objects in your squash video")
    print("- Save annotated video and detection data")
    print("- Generate AI-powered play-by-play analysis")

if __name__ == "__main__":
    main()
