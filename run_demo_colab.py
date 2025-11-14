#!/usr/bin/env python3
"""
Colab Launcher for ShifaMind Demo

Instructions:
1. Upload this file and demo1.py to Colab
2. Upload stage4_joint_best_revised.pt (model checkpoint)
3. Run this script
4. Enter your ngrok auth token when prompted
5. Click the public URL to access the demo
"""

import os
import sys
import time

print("="*70)
print("🏥 SHIFAMIND DEMO - COLAB LAUNCHER")
print("="*70)

# Install dependencies
print("\n📦 Installing dependencies...")
os.system('pip install -q streamlit pyngrok openai torch transformers faiss-cpu')

# Check if Google Drive is already mounted
print("\n📁 Checking Google Drive...")
if os.path.exists('/content/drive/MyDrive'):
    print("✅ Google Drive already mounted")
else:
    print("⚠️  Google Drive not mounted")
    print("💡 Please run this first in a Colab cell:")
    print("    from google.colab import drive")
    print("    drive.mount('/content/drive')")
    print("\nContinuing anyway (demo will work if files are in /content)...")

# Setup ngrok
print("\n🔧 Setting up ngrok tunnel...")
from pyngrok import ngrok

# Get ngrok token from user
ngrok_token = input("\n🔑 Enter your ngrok auth token (get free token from https://dashboard.ngrok.com/get-started/your-authtoken): ")

if ngrok_token.strip():
    ngrok.set_auth_token(ngrok_token.strip())
else:
    print("⚠️  No ngrok token provided. Using default (may not work)")

# Kill existing streamlit/ngrok processes
print("\n🧹 Cleaning up existing processes...")
os.system("pkill -f streamlit")
os.system("pkill -f ngrok")
time.sleep(2)

# Start Streamlit server
print("\n🚀 Starting Streamlit server...")
os.system("nohup streamlit run demo1.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &")

# Wait for server to start
print("⏳ Waiting for server to start...")
time.sleep(8)

# Create ngrok tunnel
try:
    public_url = ngrok.connect(8501)
    print("\n" + "="*70)
    print("✅ DEMO IS READY!")
    print("="*70)
    print(f"\n🌐 Public URL: {public_url}")
    print("\n👆 Click the link above to open the ShifaMind demo")
    print("\n📋 Instructions:")
    print("   1. Enter your OpenAI API key in the sidebar")
    print("   2. Click 'Load ShifaMind Model'")
    print("   3. Select a template case or enter your own")
    print("   4. Click 'Run Diagnosis Comparison'")
    print("\n💡 The demo will show ChatGPT vs ShifaMind side-by-side")
    print("="*70)

except Exception as e:
    print(f"\n❌ Error creating tunnel: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you entered a valid ngrok token")
    print("2. Get a free token from: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("3. Try restarting the Colab runtime")
