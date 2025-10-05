# 🌱 Aesculapius AI - Intelligent Household Repair Assistant

**Transform the way you approach household repairs with AI-powered sustainability**

Aesculapius AI is a desktop application that empowers users to repair instead of replace their household items through intelligent AI guidance, comprehensive parts lookup, and local technician connections. Named after the Greek god of healing, this application "heals" your broken appliances and devices while promoting environmental sustainability.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![Version](https://img.shields.io/badge/version-2.0.0-green.svg)

## 🎯 Core Mission

**Repair First, Replace Last** - Reduce electronic waste and support local businesses through intelligent DIY repair guidance.

Download: https://drive.google.com/file/d/1Jx3A5eUb4T2NTLWGkWarzA4e885BSg-i/view?usp=sharing

*May show contains Virus, click ignore and run. Not Microsft Cert." 

## ✨ Key Features
<img width="789" height="625" alt="Screenshot 2025-10-04 174059" src="https://github.com/user-attachments/assets/5cd4f91f-0602-4f85-9284-508e77cf2ee8" />
<p align="center">
  <img src="https://github.com/user-attachments/assets/b3736b7b-d7c1-4a0b-a737-87f362e7bac3" alt="Screenshot 1" width="250"/>
  <img src="https://github.com/user-attachments/assets/4705f7db-91f3-48a6-b1de-7c80df71a4bb" alt="Screenshot 2" width="250"/>
  <img src="https://github.com/user-attachments/assets/85db35b8-5015-432c-a29f-d0f8db2cf501" alt="Screenshot 3" width="250"/>
  <img src="https://github.com/user-attachments/assets/819c4f36-df0f-4315-b6cd-9eed863841b3" alt="Screenshot 4" width="250"/>
  <img src="https://github.com/user-attachments/assets/76a6639b-0695-493a-baff-72b91506a8ce" alt="Screenshot 5" width="250"/>
  <img src="https://github.com/user-attachments/assets/84ad254a-4e4c-42e9-a4fb-469a78757cbf" alt="Screenshot 6" width="250"/>
  <img src="https://github.com/user-attachments/assets/19eb2961-3605-4e7c-92a3-42fcb6489423" alt="Screenshot 7" width="250"/>
</p>




### 🤖 **AI-Powered Diagnostics**
- Local Phi-3.5-mini model for private, offline AI processing
- Conversational repair diagnostics with contextual memory
- Intelligent symptom analysis and troubleshooting guidance
- Step-by-step repair instructions tailored to your specific item

### 📦 **Smart Inventory Management**
- Comprehensive household item cataloging with AI-enhanced descriptions
- Advanced item research using real-time web scraping
- Automatic specification and feature extraction
- Repair history tracking and maintenance prediction

### 🔍 **Intelligent Parts Discovery**
- Real-time parts identification from natural language descriptions
- OEM and aftermarket compatibility checking
- Direct integration with Amazon, eBay, and specialized parts retailers
- Part number extraction and cross-referencing

### 👨‍🔧 **Local Business Support**
- Smart technician finder targeting local small businesses
- Integration with Google Maps, Yelp, NextDoor, and BBB
- Context-aware search based on repair category and location
- Community-driven recommendations and reviews

### 🔒 **Privacy-First Design**
- Complete offline AI processing - your data never leaves your device
- Local JSON-based storage with no cloud dependencies
- GDPR-compliant data handling
- No telemetry or usage tracking

## 🛠️ Technical Architecture

### **Frontend**
- **PySide6/Qt6** - Modern, responsive desktop interface
- Custom widget system with professional styling
- Real-time streaming chat interface
- Responsive grid layouts and scroll areas

### **AI Engine**
- **Microsoft Phi-3.5-mini** - 4GB quantized model for local inference
- **llama-cpp-python** - High-performance CPU-based AI processing
- **Streaming responses** - Real-time conversational experience
- **Context management** - Maintains conversation memory across sessions

### **Data Integration**
- **Web scraping** - BeautifulSoup for manufacturer website data
- **API integration** - DuckDuckGo search API for real-time information
- **Parts databases** - Automated parts lookup and verification
- **Local storage** - JSON-based item database with full backup support

### **Deployment**
- **PyInstaller** - Single-executable distribution
- **Inno Setup** - Professional Windows installer with upgrade support
- **Digital signing** - Code signing for security and trust
- **Modular architecture** - Easy maintenance and feature additions

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 (64-bit)
- 8GB RAM (recommended for AI model)
- 6GB free disk space (for AI model and application)
- Internet connection (for parts lookup and technician search)

### Installation

#### Option 1: Pre-built Installer (Recommended)
1. Download `Aesculapius-AI-Setup.exe` from [Releases](https://github.com/yourusername/aesculapius-ai/releases)
2. Run the installer as Administrator
3. Launch from Start Menu or Desktop shortcut

#### Option 2: Build from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/aesculapius-ai.git
cd aesculapius-ai

# Install dependencies
pip install -r requirements.txt

# Download the AI model
python download_model.py

# Run the application
python main.py
```

### First Run
1. **Add your first item** - Use the "Add New Item" button to catalog a household appliance
2. **Start diagnosis** - Click "🔍 Diagnose" to begin an AI-powered repair conversation
3. **Find parts** - Use the built-in parts finder to locate replacement components
4. **Connect with technicians** - Find local repair professionals for complex issues

## 📱 Usage Guide

### Adding Items
- **Smart Entry**: Enter basic info and let AI research comprehensive details
- **Manual Mode**: Full control over item specifications and descriptions
- **Bulk Import**: (Coming soon) CSV import for large inventories

### AI Diagnosis Process
1. **Symptom Description** - Describe what's wrong in natural language
2. **Guided Questioning** - AI asks targeted diagnostic questions
3. **Parts Identification** - Specific replacement parts with model numbers
4. **Repair Guidance** - Step-by-step instructions and safety tips

### Parts & Service Discovery
- **Intelligent Search** - Context-aware parts lookup with OEM compatibility
- **Price Comparison** - Multiple retailer integration for best prices
- **Local Technicians** - Small business directory with community reviews
- **Service Categories** - Specialized search for appliance, electronics, automotive, HVAC

## 🌍 Environmental Impact

**Sustainability Metrics**:
- **Waste Reduction**: Extend product lifecycles through informed repair decisions
- **Local Economy**: Support small repair businesses over big box replacements
- **Carbon Footprint**: Reduce manufacturing demand through repair-first mentality
- **Skill Building**: Empower users with DIY repair knowledge and confidence

## 🔧 Development

### Tech Stack
```
Frontend:     PySide6, Qt6, Custom CSS styling
AI/ML:        Phi-3.5-mini, llama-cpp-python, Streaming inference
Backend:      Python 3.8+, JSON storage, Threading
Integration:  BeautifulSoup, Requests, DuckDuckGo API
Build:        PyInstaller, Inno Setup, Digital signing
```

### Project Structure
```
aesculapius-ai/
├── main.py              # Main application entry point
├── download_model.py    # AI model downloader
├── parts_lookup.py      # Parts search integration
├── style.qss           # Qt stylesheet
├── requirements.txt    # Python dependencies
├── installer.iss       # Inno Setup installer script
├── models/             # AI model storage
├── cache/              # Application cache
└── dist/               # Build output directory
```

### Building
```bash
# Build executable
python -m PyInstaller main.py --name "Aesculapius-AI" --noconfirm --windowed --onedir --clean --collect-all PySide6 --collect-submodules PySide6 --copy-metadata shiboken6 --collect-all llama_cpp --collect-submodules llama_cpp

# Create installer (requires Inno Setup)
# Open installer.iss in Inno Setup Compiler and build
```

## 🤝 Contributing

We welcome contributions that advance our mission of sustainable repair culture!

### Priority Areas
- **AI Model Optimization** - Improve diagnostic accuracy and speed
- **Parts Database Expansion** - Add more manufacturer APIs and databases
- **Mobile App** - iOS/Android companion app
- **Internationalization** - Multi-language support
- **Community Features** - User-shared repair guides and tips

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft** - For the Phi-3.5-mini language model
- **llama.cpp team** - For efficient local AI inference
- **Qt/PySide6** - For the robust desktop framework
- **Repair community** - For inspiring the circular economy movement

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/aesculapius-ai/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/aesculapius-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/aesculapius-ai/discussions)
- **Email**: support@aesculapius-ai.com

---

**"Every broken appliance deserves a second chance"** ⚕️

Made with ❤️ for a sustainable future
