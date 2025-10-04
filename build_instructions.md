# Building FixMate as Standalone Executable

## Option A: Bundle with Ollama (Full Package)

### Step 1: Create Executable with PyInstaller

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --name="FixMate" ^
    --windowed ^
    --onefile ^
    --icon=icon.ico ^
    --add-data "style.qss;." ^
    main_enhanced.py
```

### Step 2: Download Ollama Installer

1. Download from: https://ollama.ai/download
2. Save as `installers/OllamaSetup.exe`

### Step 3: Pre-download Model

```bash
# Pull the model locally first
ollama pull phi

# Copy model files from:
# Windows: %LOCALAPPDATA%\Ollama\models\
# To your project: models/phi/
```

### Step 4: Create Inno Setup Script

Save as `setup.iss`:

```iss
[Setup]
AppName=FixMate
AppVersion=2.0.0
AppPublisher=Your Name
DefaultDirName={autopf}\FixMate
DefaultGroupName=FixMate
OutputDir=output
OutputBaseFilename=FixMate_Setup_v2.0.0
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Main application
Source: "dist\FixMate.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "style.qss"; DestDir: "{app}"; Flags: ignoreversion

; Ollama installer (bundled)
Source: "installers\OllamaSetup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

; Pre-downloaded model (optional - makes installer larger but faster)
Source: "models\*"; DestDir: "{localappdata}\Ollama\models"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\FixMate"; Filename: "{app}\FixMate.exe"
Name: "{autodesktop}\FixMate"; Filename: "{app}\FixMate.exe"; Tasks: desktopicon

[Run]
; Install Ollama silently
Filename: "{tmp}\OllamaSetup.exe"; Parameters: "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART"; StatusMsg: "Installing Ollama AI Engine..."; Flags: waituntilterminated

; Start Ollama service
Filename: "{localappdata}\Programs\Ollama\ollama.exe"; Parameters: "serve"; Flags: nowait runhidden; StatusMsg: "Starting Ollama service..."

; Launch FixMate
Filename: "{app}\FixMate.exe"; Description: "{cm:LaunchProgram,FixMate}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('FixMate will install Ollama AI engine (required for operation). This may take a few minutes.', mbInformation, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // Wait for Ollama to start
    Sleep(3000);
    
    // Pull the model if not bundled
    if not FileExists(ExpandConstant('{localappdata}\Ollama\models\phi')) then
    begin
      Exec(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'), 
           'pull phi', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;
```

### Step 5: Build Installer

```bash
# Compile with Inno Setup
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss
```

---

## Option B: Lightweight Installer (Download During Install)

For a smaller installer that downloads Ollama during installation:

```iss
[Setup]
AppName=FixMate
AppVersion=2.0.0
OutputBaseFilename=FixMate_Setup_Lite_v2.0.0

[Files]
Source: "dist\FixMate.exe"; DestDir: "{app}"
Source: "style.qss"; DestDir: "{app}"

[Code]
var
  DownloadPage: TDownloadWizardPage;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if Progress = ProgressMax then
    Log(Format('Successfully downloaded %s', [FileName]));
  Result := True;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then begin
    DownloadPage.Clear;
    DownloadPage.Add('https://ollama.ai/download/OllamaSetup.exe', 'OllamaSetup.exe', '');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        Result := True;
      except
        if DownloadPage.AbortedByUser then
          Log('Aborted by user.')
        else
          SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);
        Result := False;
      end;
    finally
      DownloadPage.Hide;
    end;
  end else
    Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // Install Ollama
    Exec(ExpandConstant('{tmp}\OllamaSetup.exe'), '/VERYSILENT', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    
    // Start Ollama
    Exec(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'), 'serve', '', SW_HIDE, ewNoWait, ResultCode);
    Sleep(3000);
    
    // Pull model
    Exec(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'), 'pull phi', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
```

---

## Option C: Use Embedded LLM (No Ollama Required)

Replace Ollama with **llama-cpp-python** for a truly standalone app:

### 1. Modify requirements.txt:

```txt
PySide6>=6.5.0
requests>=2.31.0
llama-cpp-python>=0.2.0  # Replace Ollama
```

### 2. Download GGUF model:

```bash
# Download a small GGUF model (e.g., Phi-2)
# From: https://huggingface.co/TheBloke/phi-2-GGUF
# Save to: models/phi-2.Q4_K_M.gguf
```

### 3. Modify main_enhanced.py:

```python
from llama_cpp import Llama

# Load model at startup
llm = Llama(model_path="models/phi-2.Q4_K_M.gguf", n_ctx=2048)

def identify_item_from_description(user_input: str) -> dict:
    response = llm(prompt, max_tokens=256, temperature=0.7)
    # Parse response...
```

### 4. Bundle everything:

```bash
pyinstaller --add-data "models/phi-2.Q4_K_M.gguf;models" main_enhanced.py
```

**Result**: Single ~2GB executable, no external dependencies!

---

## Recommended Approach

**For most users**: **Option A** (Bundle Ollama + Model)
- Pros: Works offline immediately, familiar Ollama interface
- Cons: Large installer (~2-3GB)

**For advanced users**: **Option C** (Embedded GGUF)
- Pros: True standalone, no services, portable
- Cons: More complex code changes

---

## Testing Your Installer

1. Test on a clean Windows VM
2. Verify Ollama installs correctly
3. Check model downloads successfully
4. Ensure app launches and connects to Ollama

## File Size Estimates

- FixMate.exe: ~50MB (with PySide6)
- Ollama installer: ~200MB
- Phi model: ~1.6GB
- **Total installer**: ~1.9GB (compressed with LZMA)

---

## Alternative: Portable Version

Create a portable ZIP instead:

```
FixMate_Portable/
├── FixMate.exe
├── style.qss
├── ollama/
│   └── ollama.exe
└── models/
    └── phi/
```

Users extract and run `start.bat`:

```batch
@echo off
start /B ollama\ollama.exe serve
timeout /t 3
FixMate.exe
```
