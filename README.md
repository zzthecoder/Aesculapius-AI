# 🛠️ FixMate: AI-Powered Repair Assistant

A desktop repair assistant app that helps you track household items, diagnose problems using a local LLM, and find replacement parts.

## ✨ Features

- **Item Management**: Track household items with descriptions and usage history
- **AI-Powered Diagnosis**: Interactive chat interface with local LLM for problem diagnosis
- **Smart Recommendations**: Get repair suggestions, part numbers, and fix instructions
- **Parts Lookup**: Integrated web search for replacement parts and repair guides
- **Offline-First**: Works entirely offline with optional web access
- **Dark Mode UI**: Clean, modern interface with dark theme
- **Privacy-Focused**: All data stored locally, no cloud dependencies

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **Ollama** running locally with a model pulled

### Setup Ollama

```bash
# Install Ollama (visit https://ollama.ai for installation)

# Start Ollama service
ollama serve

# Pull a model (in a new terminal)
ollama pull mistral
```

### Install FixMate

```bash
# Clone or download the project
cd Fixmate

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Run the enhanced version with full features
python main_enhanced.py

# Or run the original version
python main.py
```

## 📖 User Guide

### Adding Items

1. Click **"Add New Item"** on the dashboard
2. Describe your item (e.g., "My 5-year-old Samsung washing machine, model WF45R6300AV")
3. The AI will extract the name, description, and age automatically
4. Item appears on your dashboard

### Diagnosing Problems

1. Click **"Diagnose Problem"** on any item card
2. Describe the issue in the chat interface
3. The AI assistant will:
   - Ask follow-up questions
   - Deduce potential causes
   - Suggest replacement parts
   - Provide repair instructions
4. Click **"Save Diagnosis"** to save the conversation to item history

### Finding Parts

The `parts_lookup.py` module provides:
- **iFixit repair guides** (free, no API key needed)
- **Web search results** via DuckDuckGo
- **Shopping links** for Amazon, eBay, AliExpress, Google Shopping

```python
# Example usage in Python
from parts_lookup import search_parts_comprehensive

results = search_parts_comprehensive("Coffee Maker", "heating element")
print(results['guides'])  # iFixit guides
print(results['shopping_links'])  # Direct shopping URLs
```

## 🏗️ Architecture

### Tech Stack

- **Frontend**: PySide6 (Qt for Python)
- **LLM**: Ollama (local inference)
- **Storage**: JSON file-based (data.json)
- **Styling**: QSS (Qt Style Sheets)
- **Web Integration**: requests, optional scrapy

### File Structure

```
Fixmate/
├── main.py                 # Original version
├── main_enhanced.py        # Enhanced version with full features
├── parts_lookup.py         # Web scraping & parts search
├── style.qss              # Dark mode stylesheet
├── requirements.txt       # Python dependencies
├── data.json             # Local data storage (auto-generated)
└── README.md             # This file
```

## 🔧 Configuration

### Change LLM Model

Edit the `OLLAMA_MODEL` constant in `main_enhanced.py`:

```python
OLLAMA_MODEL = "mistral"  # Options: mistral, llama2, phi-2, etc.
```

### Customize Ollama URL

If Ollama is running on a different port:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
```

## 🌐 Web Integration

### iFixit API

The app uses iFixit's public API (no key required) for repair guides.

### Scrapy Integration

For advanced scraping, see `parts_lookup.py`. A Scrapy spider template is included.

### Farfalle Integration

[Farfalle](https://github.com/rashadphz/farfalle) is an AI-powered search engine. To integrate:

1. Run farfalle service separately
2. Add API calls to `parts_lookup.py`
3. Configure endpoint in constants

## 📊 Data Model

Items are stored as:

```json
{
  "id": "item_1696204800000",
  "name": "Coffee Maker",
  "description": "5-cup drip machine, model CM-245",
  "yearsUsed": 3,
  "diagnosisHistory": [
    {
      "date": "2025-10-02 00:35",
      "issuesReported": "Won't brew",
      "suggestedParts": ["Heating coil #HC245"],
      "outcome": "Diagnosis completed"
    }
  ]
}
```

## 🎨 Customization

### Modify Dark Theme

Edit `style.qss` to customize colors, fonts, and styling.

### Add Custom Widgets

All UI components are in the main file. Create new widget classes as needed.

## 🐛 Troubleshooting

### "Could not connect to Ollama"

- Ensure Ollama is running: `ollama serve`
- Check the URL in the code matches your Ollama instance
- Verify a model is pulled: `ollama list`

### "LLM response did not contain all required fields"

- Try a different model (e.g., `mistral` instead of `llama2`)
- Increase timeout in the code if responses are slow

### Styling not applied

- Ensure `style.qss` exists in the same directory
- Check for syntax errors in the QSS file

## 🚧 Future Enhancements

- [ ] Export diagnosis reports to PDF
- [ ] Image upload for visual diagnosis
- [ ] Integration with manufacturer APIs
- [ ] Multi-language support
- [ ] Cloud sync (optional)
- [ ] Mobile companion app

## 📝 License

This project is provided as-is for educational and personal use.

## 🤝 Contributing

Feel free to fork, modify, and improve! Suggestions welcome.

## 📧 Support

For issues or questions, please check:
- Ollama documentation: https://ollama.ai
- PySide6 documentation: https://doc.qt.io/qtforpython/
- iFixit API: https://www.ifixit.com/api/2.0/doc

---

**Built with ❤️ for sustainability and right-to-repair**
