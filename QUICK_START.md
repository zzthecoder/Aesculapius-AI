# 🚀 FixMate Quick Start (Embedded Version)

## What Changed?

✅ **No Ollama needed!** - Uses embedded llama-cpp-python
✅ **Truly standalone** - Everything in one package
✅ **Easy to distribute** - Single EXE with model included

---

## Get Started in 3 Steps

### 1️⃣ Install Dependencies (IMPORTANT!)

**Option A: Use Install Script (Recommended)**

```bash
# PowerShell
.\install.ps1

# OR Command Prompt
install.bat
```

**Option B: Manual Install**

```bash
# MUST include the extra index URL for pre-built wheels!
pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

⚠️ **DO NOT** run just `pip install -r requirements.txt` - it will fail without build tools!

### 2️⃣ Download Model

```bash
python download_model.py
```

### 3️⃣ Run the App

```bash
python main_enhanced.py
```

---

## Build Standalone EXE

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable with embedded model
pyinstaller --name="FixMate" --windowed --onefile --add-data "style.qss;." --add-data "models/phi-2.Q4_K_M.gguf;models" main_enhanced.py

# Your EXE is ready!
# Location: dist/FixMate.exe (~1.7GB)
```

---

## What's Included?

- ✅ **main_enhanced.py** - Updated to use llama-cpp-python
- ✅ **download_model.py** - Automatic model downloader
- ✅ **requirements.txt** - Updated dependencies
- ✅ **SETUP_EMBEDDED.md** - Complete setup guide
- ✅ **build_instructions.md** - Inno Setup packaging guide

---

## File Structure

```
Fixmate/
├── main_enhanced.py          # Main app (uses embedded LLM)
├── download_model.py          # Model downloader
├── requirements.txt           # Dependencies (with llama-cpp-python)
├── style.qss                  # Dark theme
├── parts_lookup.py            # Web scraping utilities
├── models/                    # Model folder (created by download script)
│   └── phi-2.Q4_K_M.gguf     # AI model (~1.6GB)
└── dist/                      # Built executables (after PyInstaller)
    └── FixMate.exe            # Standalone app
```

---

## Advantages

| Feature | Ollama Version | Embedded Version |
|---------|---------------|------------------|
| Setup | Install Ollama + Pull model | Download model file |
| Dependencies | External service | Self-contained |
| Portability | Requires Ollama | Fully portable |
| Distribution | Complex | Single EXE |
| Startup | Fast | ~10s (model loading) |
| Size | Small app + Ollama | ~1.7GB single file |

---

## Next Steps

1. **Test the app**: Run `python main_enhanced.py`
2. **Build EXE**: Follow build instructions above
3. **Create installer**: Use Inno Setup (see `build_instructions.md`)
4. **Distribute**: Share the installer with users!

---

## Need Help?

- **Setup issues**: See `SETUP_EMBEDDED.md`
- **Building EXE**: See `build_instructions.md`
- **Model not found**: Run `python download_model.py`

---

**You're all set! 🎉**
