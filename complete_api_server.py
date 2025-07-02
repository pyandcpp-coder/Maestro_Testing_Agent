#!/usr/bin/env python3
"""
Complete setup script for mobile UI test automation server.
This script handles all dependencies, model downloading, and server startup.
Run this single file to get everything working.
"""

import subprocess
import os
import sys
import zipfile
import requests
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def install_system_dependencies():
    """Install system packages"""
    print("🚀 Installing system dependencies...")
    run_command("apt-get update && apt-get install -y unzip wget curl", "System package installation")

def install_python_dependencies():
    """Install Python packages"""
    print("📦 Installing Python dependencies...")
    packages = [
        "fastapi",
        "uvicorn[standard]", 
        "transformers",
        "peft",
        "torch",
        "python-dotenv",
        "google-generativeai",
        "gdown",
        "pydantic",
        "huggingface_hub",
        "accelerate"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")

def download_adapter():
    """Download and extract the LoRA adapter"""
    print("📥 Downloading LoRA adapter...")
    
    os.makedirs("lora_adapter", exist_ok=True)
    zip_path = "lora_adapter/adapter.zip"

    # Download the adapter zip file if it doesn't exist
    if not os.path.exists(zip_path):
        download_cmd = "python -c \"import gdown; gdown.download('https://drive.google.com/uc?id=1-olLOo2F3LuMU7twhLCAtJBHWv1FDrQe', '{}', quiet=False)\"".format(zip_path)
        if not run_command(download_cmd, "Downloading adapter from Google Drive"):
            print("❌ Failed to download adapter.")
            return

    # Extract the zip file
    try:
        print("📦 Extracting adapter...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("lora_adapter/")
        print("✅ Adapter extracted successfully")
    except Exception as e:
        print(f"❌ Failed to extract adapter: {e}")


def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    # Create .env file with API key
    env_content = ''' # Environment variables for Mobile UI Test Automation Server'''
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment file created")

def setup_huggingface_login():
    """Login to Hugging Face"""
    print("🤗 Setting up Hugging Face authentication...")
    try:
        from huggingface_hub import login
        login("your-huggingface-token-here")  # Replace with your Hugging Face token
        print("✅ Hugging Face login successful")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting the server...")
    
    # Server code
    server_code = '''
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import google.generativeai as genai

# === Load .env ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
genai.configure(api_key=GOOGLE_API_KEY)
analyst_model = genai.GenerativeModel("gemini-1.5-flash")

# === Constants ===
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "lora_adapter/checkpoint-272_new_workspace_full_version"

# === FastAPI Setup ===
app = FastAPI(title="Mobile UI Test Automation API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Global variables ===
model = None
tokenizer = None

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=False)
    model = model.merge_and_unload()
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("✅ Model loaded on GPU")
    else:
        print("⚠️ GPU not available, using CPU")
    
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model and tokenizer loaded successfully")

# === Request Schemas ===
class GenerateRequest(BaseModel):
    command: str

class AnalyzeRequest(BaseModel):
    command: str
    yaml: str
    stdout: str
    stderr: str
    success: bool

# === Health Check ===
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Mobile UI Test Automation API is running"}

@app.get("/health")
def detailed_health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# === Generate YAML Endpoint ===
@app.post("/generate-yaml")
def generate_yaml(req: GenerateRequest):
    if model is None or tokenizer is None:
        return {"error": "Model not loaded. Please wait for initialization."}
    
    prompt = f"<s>[INST] {req.command} [/INST]"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    yaml_output = result.split("[/INST]")[-1].strip()
    return {"yaml": yaml_output}

# === Analyze Test Report Endpoint ===
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        status = "PASSED" if req.success else "FAILED"
        prompt = f"""
        Analyze this mobile UI test run and give a QA-style report:
        Command: {req.command}
        Status: {status}
        --- YAML ---
        {req.yaml}
        --- STDOUT ---
        {req.stdout}
        --- STDERR ---
        {req.stderr}
        """

        response = analyst_model.generate_content(prompt)
        return {"report": response.text}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting up the server...")
    load_model()

if __name__ == "__main__":
    print("🚀 Starting Mobile UI Test Automation Server...")
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )
    '''
    
    # Write server code to file
    with open('api_server.py', 'w') as f:
        f.write(server_code)
    
    print("✅ Server code written to api_server.py")

def main():
    """Main setup function"""
    print("🎯 Starting complete setup for Mobile UI Test Automation Server")
    print("=" * 60)
    
    # Step 1: Install system dependencies
    install_system_dependencies()
    
    # Step 2: Install Python dependencies  
    install_python_dependencies()
    
    # Step 3: Setup environment
    setup_environment()
    
    # Step 4: Setup Hugging Face login
    setup_huggingface_login()
    
    # Step 5: Download adapter
    download_adapter()
    
    # Step 6: Create server file
    start_server()
    
    print("=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    
    # Ask if user wants to start server immediately
    response = input("🚀 Do you want to start the server now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("🚀 Starting server...")
        os.system("python api_server.py")

if __name__ == "__main__":
    main()