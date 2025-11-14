"""
🏥 SHIFAMIND DEMO - SIMPLE COLAB SETUP
Copy and paste these cells into Google Colab notebook
"""

# ==============================================================================
# CELL 1: Mount Drive and Install Dependencies
# ==============================================================================
"""
Run this cell first (takes ~1 minute)
"""

# Mount Google Drive (if needed for model checkpoint)
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
print("📦 Installing dependencies...")
!pip install -q streamlit pyngrok openai torch transformers faiss-cpu

print("✅ Setup complete!")

# ==============================================================================
# CELL 2: Upload Files
# ==============================================================================
"""
Upload these 3 files using the file browser (left sidebar):
1. demo1.py
2. stage4_joint_best_revised.pt
3. (optional) Copy from Drive if you have it there

OR if files are in Google Drive:
"""

# If your files are in Google Drive, copy them:
# !cp /content/drive/MyDrive/ShifaMind/demo1.py /content/
# !cp /content/drive/MyDrive/ShifaMind/stage4_joint_best_revised.pt /content/

# Verify files exist
import os
print("✅ Checking files...")
for f in ['demo1.py', 'stage4_joint_best_revised.pt']:
    if os.path.exists(f):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} - Please upload!")

# ==============================================================================
# CELL 3: Start Streamlit with Ngrok
# ==============================================================================
"""
Run this cell to start the demo
You'll need your ngrok auth token: https://dashboard.ngrok.com/get-started/your-authtoken
"""

import os
import time
from pyngrok import ngrok

# Get ngrok token (replace with yours or input)
NGROK_TOKEN = input("Enter your ngrok auth token: ")

# Set auth token
ngrok.set_auth_token(NGROK_TOKEN)

# Kill any existing processes
!pkill -f streamlit
!pkill -f ngrok
time.sleep(2)

# Start Streamlit in background
print("🚀 Starting Streamlit...")
!nohup streamlit run demo1.py --server.port 8501 --server.headless true > /dev/null 2>&1 &

# Wait for server to start
time.sleep(8)

# Create ngrok tunnel
public_url = ngrok.connect(8501)

print("\n" + "="*70)
print("✅ DEMO IS READY!")
print("="*70)
print(f"\n🌐 Public URL: {public_url}")
print("\n👆 Click the link above to open the demo")
print("\n📋 Next steps in the demo:")
print("   1. Enter your OpenAI API key in the sidebar")
print("   2. Click 'Load ShifaMind Model' (wait 30 sec)")
print("   3. Select a template case")
print("   4. Click 'Run Diagnosis Comparison'")
print("\n💡 Show this to the doctor!")
print("="*70)

# ==============================================================================
# OPTIONAL CELL 4: Check Logs (if something goes wrong)
# ==============================================================================
"""
If the demo isn't working, run this to see Streamlit logs:
"""

!tail -n 50 streamlit.log

# ==============================================================================
# OPTIONAL CELL 5: Restart Demo (if needed)
# ==============================================================================
"""
If you need to restart the demo:
"""

!pkill -f streamlit
!pkill -f ngrok
time.sleep(2)

!nohup streamlit run demo1.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
time.sleep(8)

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"🌐 New URL: {public_url}")
