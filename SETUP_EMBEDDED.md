# FixMate Setup Guide (Embedded LLM Version)

## ✅ No Ollama Required!

This version uses **llama-cpp-python** with embedded GGUF models. Everything runs locally with no external services.

---

## 📥 Step 1: Install Dependencies

```bash
# Activate your virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies (this will take a few minutes)
pip install -r requirements.txt
```

**Note**: `llama-cpp-python` will compile C++ code. This is normal and may take 5-10 minutes.

---

## 📦 Step 2: Download the Model

### Option A: Automatic Download (Recommended)

Run this Python script to download the model automatically:

```python
# download_model.py
import requests
import os

MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
MODEL_PATH = "models/phi-2.Q4_K_M.gguf"

os.makedirs("models", exist_ok=True)

print("Downloading Phi-2 model (~1.6GB)...")
print("This may take a few minutes depending on your internet speed.")

response = requests.get(MODEL_URL, stream=True)
total_size = int(response.headers.get('content-length', 0))
downloaded = 0

with open(MODEL_PATH, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            percent = (downloaded / total_size) * 100
            print(f"\rProgress: {percent:.1f}%", end='')

print("\n✅ Model downloaded successfully!")
```

Run it:
```bash
python download_model.py
```

### Option B: Manual Download

1. Visit: https://huggingface.co/TheBloke/phi-2-GGUF
2. Download: **phi-2.Q4_K_M.gguf** (~1.6GB)
3. Create folder: `models/`
4. Place the file in: `models/phi-2.Q4_K_M.gguf`

### Alternative Models

You can use other GGUF models:

| Model | Size | Quality | Download |
|-------|------|---------|----------|
| **Phi-2 Q4** | 1.6GB | Good | Recommended |
| Phi-2 Q5 | 2.0GB | Better | Higher quality |
| Mistral Q4 | 4.1GB | Best | Slower but smarter |
| TinyLlama Q4 | 637MB | Basic | Fastest, less accurate |

To use a different model, update `MODEL_PATH` in `main_enhanced.py`:

```python
MODEL_PATH = "models/your-model-name.gguf"
```

---

## 🚀 Step 3: Run the App

```bash
python main_enhanced.py
```

The first launch will take ~10 seconds to load the model into memory.

---

## 📦 Building Standalone Executable

### Step 1: Install PyInstaller

```bash
pip install pyinstaller
```

### Step 2: Create Executable

```bash
pyinstaller --name="FixMate" ^
    --windowed ^
    --onefile ^
    --add-data "style.qss;." ^
    --add-data "models/phi-2.Q4_K_M.gguf;models" ^
    main_enhanced.py
```

This creates a **single executable** with the model embedded!

**Output**: `dist/FixMate.exe` (~1.7GB)

### Step 3: Test the Executable

```bash
cd dist
.\FixMate.exe
```

---

## 🎁 Creating Installer with Inno Setup

### Simple Inno Setup Script

Save as `setup_embedded.iss`:

```iss
[Setup]
AppName=FixMate
AppVersion=2.0.0
AppPublisher=Your Name
DefaultDirName={autopf}\FixMate
DefaultGroupName=FixMate
OutputDir=output
OutputBaseFilename=FixMate_Setup_v2.0.0_Embedded
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Single executable with embedded model
Source: "dist\FixMate.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\FixMate"; Filename: "{app}\FixMate.exe"
Name: "{autodesktop}\FixMate"; Filename: "{app}\FixMate.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FixMate.exe"; Description: "{cm:LaunchProgram,FixMate}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('FixMate is a standalone repair assistant with embedded AI. No internet or external services required!', mbInformation, MB_OK);
end;
```

### Build the Installer

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup_embedded.iss
```

**Output**: `output/FixMate_Setup_v2.0.0_Embedded.exe` (~1.8GB compressed)

---

## 🎯 Advantages of Embedded Version

✅ **No Ollama required** - Everything in one package
✅ **Truly portable** - Copy and run anywhere
✅ **No services** - No background processes
✅ **Offline-first** - Works without internet
✅ **Easy distribution** - Single installer or executable
✅ **Faster startup** - No API calls, direct inference

---

## ⚙️ Performance Tips

### CPU Optimization

Edit `main_enhanced.py` to adjust threads:

```python
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,  # Increase for more CPU cores
    verbose=False
)
```

### GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
# Install CUDA version
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Then update the code:

```python
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=35,  # Offload layers to GPU
    verbose=False
)
```

---

## 🐛 Troubleshooting

### "Model file not found"

Make sure the model is at: `models/phi-2.Q4_K_M.gguf`

### "Failed to load shared library"

Reinstall llama-cpp-python:
```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --force-reinstall
```

### Slow responses

- Use a smaller model (TinyLlama)
- Increase `n_threads` in the code
- Reduce `max_tokens` in generation

### Out of memory

- Use a smaller quantization (Q4 instead of Q5)
- Reduce `n_ctx` (context window)
- Close other applications

---

## 📊 Model Comparison

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| TinyLlama Q4 | 637MB | 1GB | ⚡⚡⚡ | ⭐⭐ |
| **Phi-2 Q4** | 1.6GB | 3GB | ⚡⚡ | ⭐⭐⭐⭐ |
| Phi-2 Q5 | 2.0GB | 4GB | ⚡ | ⭐⭐⭐⭐⭐ |
| Mistral Q4 | 4.1GB | 6GB | ⚡ | ⭐⭐⭐⭐⭐ |

---

## 🎉 You're Done!

Your FixMate app is now:
- ✅ Fully standalone
- ✅ No external dependencies
- ✅ Ready to distribute
- ✅ Works completely offline

Enjoy your AI-powered repair assistant! 🛠️
