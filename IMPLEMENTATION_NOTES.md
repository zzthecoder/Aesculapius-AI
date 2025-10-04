# Implementation Notes

## What's Been Implemented

### ✅ Core Features (Completed)

1. **Dark Mode Styling** (`style.qss`)
   - Complete dark theme with modern UI
   - Custom styling for buttons, cards, chat messages
   - Proper contrast and readability

2. **LLM Diagnosis System** (`main_enhanced.py`)
   - Streaming chat interface with Ollama
   - Real-time response rendering
   - Conversation history tracking
   - Save diagnosis to item history

3. **Parts Lookup Module** (`parts_lookup.py`)
   - iFixit API integration (repair guides)
   - DuckDuckGo search (no API key needed)
   - Direct shopping links (Amazon, eBay, AliExpress, Google)
   - Comprehensive search function

4. **Data Persistence**
   - JSON-based storage
   - Diagnosis history saved per item
   - Proper serialization/deserialization

## Files Overview

### Main Application Files

- **`main.py`**: Original version (basic functionality)
- **`main_enhanced.py`**: ✨ **USE THIS** - Full-featured version with chat diagnosis
- **`style.qss`**: Dark mode stylesheet
- **`parts_lookup.py`**: Web scraping and parts search utilities
- **`requirements.txt`**: Python dependencies
- **`README.md`**: Complete user documentation

### Auto-Generated Files

- **`data.json`**: Created automatically when you add items

## How to Use

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull mistral
```

### 2. Run the Enhanced App

```bash
python main_enhanced.py
```

### 3. Test Parts Lookup (Optional)

```bash
python parts_lookup.py
```

## Key Differences: main.py vs main_enhanced.py

| Feature | main.py | main_enhanced.py |
|---------|---------|------------------|
| Item Management | ✅ | ✅ |
| AI Item Identification | ✅ | ✅ |
| Diagnosis Page | ❌ Placeholder only | ✅ Full chat interface |
| LLM Streaming | ❌ | ✅ |
| Save Diagnosis History | ❌ | ✅ |
| Chat UI | ❌ | ✅ |
| Parts Lookup Integration | ❌ | 🔧 Ready (see below) |

## Next Steps: Integrating Parts Lookup into UI

To add a "Find Parts" button in the diagnosis page, add this to `main_enhanced.py`:

### Option 1: Add to DiagnosisPage class

```python
# In DiagnosisPage.__init__, add a button:
self.find_parts_button = QPushButton("🔍 Find Parts Online")
self.find_parts_button.clicked.connect(self.open_parts_search)
input_layout.addWidget(self.find_parts_button)

# Add this method to DiagnosisPage:
def open_parts_search(self):
    if not self.current_item:
        return
    
    from parts_lookup import search_parts_comprehensive
    import webbrowser
    
    # Extract part description from conversation
    part_desc = "replacement parts"
    for msg in self.conversation_history:
        if "replace" in msg['content'].lower():
            part_desc = msg['content']
            break
    
    results = search_parts_comprehensive(
        self.current_item.name, 
        part_desc
    )
    
    # Show results in a dialog or open browser
    if results['shopping_links']:
        # Open first shopping link
        webbrowser.open(results['shopping_links'][0]['url'])
```

### Option 2: Automatic Part Suggestions

The LLM already suggests parts in the conversation. You can extract them and show clickable links.

## Farfalle Integration

[Farfalle](https://github.com/rashadphz/farfalle) is a separate AI-powered search service. To integrate:

1. **Run farfalle locally** (follow their setup instructions)
2. **Add to parts_lookup.py**:

```python
def search_with_farfalle(query: str) -> List[Dict]:
    """Search using farfalle API"""
    try:
        response = requests.post(
            "http://localhost:3000/api/search",  # Adjust port
            json={"query": query}
        )
        return response.json()
    except Exception as e:
        print(f"Farfalle error: {e}")
        return []
```

3. **Call from diagnosis page** when user asks for parts

## Scrapy Advanced Usage

For custom web scraping:

1. Create a Scrapy project:
```bash
scrapy startproject parts_scraper
```

2. Use the spider template from `parts_lookup.py`

3. Run spiders programmatically or via CLI

## Architecture Notes

### Why PySide6 instead of React?

The documentation mentioned React, but this implementation uses **PySide6 (Qt)** because:
- ✅ True desktop app (no browser needed)
- ✅ Better offline support
- ✅ Native OS integration
- ✅ Easier local LLM integration
- ✅ No web server required

Both approaches are valid. This is a **desktop-native** alternative.

### Data Flow

```
User Input → DiagnosisPage → DiagnosisWorker (Thread)
                                      ↓
                              Ollama LLM (HTTP)
                                      ↓
                              Streaming Response
                                      ↓
                              ChatMessage Widget
                                      ↓
                              Conversation History
                                      ↓
                              Save to Item.diagnosisHistory
                                      ↓
                              JSON File (data.json)
```

## Performance Considerations

- **LLM Streaming**: Runs in separate thread to avoid UI freezing
- **Item Identification**: Background thread with loading cursor
- **Parts Lookup**: Async-ready (can be moved to threads if needed)

## Security Notes

- All data stored locally (no cloud)
- No API keys required for basic functionality
- Ollama runs locally (no data sent to external servers)
- Web scraping respects robots.txt (when using scrapy)

## Customization Tips

### Change Colors

Edit `style.qss`:
```css
QFrame#ItemCard {
    background-color: #YOUR_COLOR;
}
```

### Add More LLM Models

```python
# In main_enhanced.py
OLLAMA_MODEL = "llama2"  # or "phi-2", "codellama", etc.
```

### Custom Prompts

Edit the `system_context` in `stream_diagnosis_response()` function.

## Known Limitations

1. **iFixit API**: Public endpoint, rate limits may apply
2. **DuckDuckGo**: Instant answers only, not full search results
3. **No Image Support**: Text-only diagnosis (can be added with base64 encoding)
4. **Single User**: No multi-user support (by design)

## Future Enhancement Ideas

- [ ] Add camera/screenshot capture for visual diagnosis
- [ ] Export diagnosis to PDF report
- [ ] Voice input for hands-free operation
- [ ] Integration with manufacturer warranty databases
- [ ] Community sharing of diagnosis solutions (optional)
- [ ] Barcode/QR code scanning for model numbers

## Troubleshooting

### Import Errors

```bash
# If PySide6 fails to install:
pip install PySide6 --upgrade

# If requests fails:
pip install requests --upgrade
```

### Ollama Connection Issues

```bash
# Check if Ollama is running:
curl http://localhost:11434/api/tags

# Restart Ollama:
# Windows: Restart from system tray
# Mac/Linux: killall ollama && ollama serve
```

### UI Not Updating

- Check if QTimer is working (might need QApplication.processEvents())
- Verify signals are connected properly
- Use `@Slot()` decorators on all slot methods

## Contributing

To extend this project:

1. Keep `main.py` as the simple version
2. Add features to `main_enhanced.py`
3. Create separate modules for complex features
4. Update `requirements.txt` if adding dependencies
5. Document changes in this file

---

**Status**: ✅ All core features implemented and ready to use!

**Recommended**: Start with `main_enhanced.py` for the full experience.
