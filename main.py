# ==============================================================================
# Asclepius AI - Intelligent Repair Assistant (Embedded Edition)
#
# main.py: Embedded-version UI and LLM workflow for repair diagnostics
# Author: AI Assistant
# Updated: October 4, 2025
#
# To Run:
# 1. Ensure the embedded model is downloaded via `python download_model.py`
# 2. Install dependencies: `pip install -r requirements.txt`
# 3. Launch the app with `python main.py`
# ==============================================================================

import sys
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

# --- Third-party libraries ---
import requests
from llama_cpp import Llama
from PySide6.QtCore import (Qt, QThread, Signal, Slot, QSize, QTimer)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDialog, QTextEdit, QMessageBox, QGridLayout, QScrollArea,
    QFrame, QStackedWidget, QSizePolicy, QLineEdit, QGroupBox, QComboBox
)
from PySide6.QtGui import QCursor
import os

# ==============================================================================
# --- Constants ---
# ==============================================================================
# Model Configuration (Embedded GGUF)
MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_M.gguf"  # Phi-3.5-mini model
DATA_FILE = "data.json"
APP_VERSION = "2.0.0-Embedded"

# Global LLM instance (loaded once at startup)
llm = None


def extract_json_snippet(text: str, start_char: str) -> str:
    """Extract the first balanced JSON snippet starting with start_char."""
    if start_char not in {"{", "["}:
        raise ValueError("start_char must be '{' or '['")

    end_char = "}" if start_char == "{" else "]"
    start_idx = text.find(start_char)
    if start_idx == -1:
        return ""

    depth = 0
    in_string = False
    escape = False

    for idx in range(start_idx, len(text)):
        ch = text[idx]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx:idx + 1]

    return ""

# ==============================================================================
# --- Data Models ---
# ==============================================================================
@dataclass
class DiagnosisEntry:
    date: str
    issuesReported: str
    suggestedParts: List[str]
    outcome: Optional[str] = None

@dataclass
class Item:
    name: str
    description: str
    yearsUsed: int
    id: str = field(default_factory=lambda: f"item_{int(time.time() * 1000)}")
    diagnosisHistory: List[DiagnosisEntry] = field(default_factory=list)
    # Enhanced fields for comprehensive item data
    brand: str = ""
    model: str = ""
    category: str = ""
    specifications: List[str] = field(default_factory=list)
    common_issues: List[str] = field(default_factory=list)
    parts_info: List[str] = field(default_factory=list)
    manual_links: List[str] = field(default_factory=list)
    year_range: str = ""
    price_range: str = ""
    model_numbers: List[str] = field(default_factory=list)
    part_numbers: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    dimensions: str = ""
    warranty: str = ""
    energy_rating: str = ""
    additional_details: dict = field(default_factory=dict)

# ==============================================================================
# --- LLM Service (Embedded llama.cpp) ---
# ==============================================================================
def load_llm():
    """Load the LLM model once at startup."""
    global llm
    if llm is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}\n\n"
                "Please run: python download_model.py"
            )
        print(f"Loading model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,  # Increased context window for Phi-3.5-mini
            n_threads=4,  # CPU threads
            n_gpu_layers=0,  # Use CPU only for compatibility
            verbose=False,
            chat_format="chatml"  # Phi-3.5-mini uses ChatML format
        )
        print("✅ Model loaded successfully!")
    return llm

def search_item_online(user_input: str) -> list:
    """Search the web for real products matching the description."""
    import requests
    import re

    results: List[dict] = []
    encoded_query = quote_plus(user_input)

    def add_result(text: str, url: str = "", source: str = "duckduckgo"):
        text = text.strip()
        if not text:
            return
        results.append({
            'source': source,
            'text': text,
            'url': url
        })

    try:
        response = requests.get(
            f"https://api.duckduckgo.com/?q={encoded_query}&format=json",
            timeout=5  # Reduced timeout for faster response
        )
        response.raise_for_status()
        data = response.json()

        for topic in data.get('RelatedTopics', [])[:3]:  # Limit to 3 for speed
            if not isinstance(topic, dict):
                continue
            text = topic.get('Text', '')
            url = topic.get('FirstURL', '')
            add_result(text, url)

    except Exception as e:
        print(f"Web search error: {e}")
        # Continue with fallback instead of failing

    if len(results) < 3:
        keyword_matches = re.findall(r'\b[A-Z][A-Za-z0-9\-]+\b', user_input)
        if keyword_matches:
            add_result(" ".join(keyword_matches[:3]), '', source="extracted")

    if not results:
        add_result(user_input, '', source="fallback")

    return results[:3]


def advanced_manufacturer_research(brand: str, model: str, item_type: str) -> dict:
    """Perform deep web scraping from manufacturer websites and spec databases."""
    import requests
    from bs4 import BeautifulSoup
    import re
    import time
    
    research_data = {
        'specifications': [],
        'common_issues': [],
        'parts_info': [],
        'manual_links': [],
        'model_numbers': [],
        'part_numbers': [],
        'year_range': '',
        'price_range': '',
        'features': [],
        'dimensions': '',
        'warranty': '',
        'energy_rating': ''
    }
    
    if not brand or not model:
        return research_data
    
    # Manufacturer-specific website scraping
    manufacturer_sites = {
        'samsung': [
            f"https://www.samsung.com/us/support/owners/product/{model.lower()}",
            f"https://www.samsung.com/us/home-appliances/{item_type.lower().replace(' ', '-')}/{model.lower()}",
            f"https://www.samsung.com/us/support/troubleshooting/{model.lower()}"
        ],
        'lg': [
            f"https://www.lg.com/us/support/product-help/CT10000018-{model.upper()}",
            f"https://www.lg.com/us/{item_type.lower().replace(' ', '-')}/{model.lower()}",
            f"https://gscs.lge.com/model/{model.upper()}"
        ],
        'whirlpool': [
            f"https://www.whirlpool.com/support/product/{model.upper()}",
            f"https://www.whirlpool.com/{item_type.lower().replace(' ', '-')}/{model.lower()}"
        ],
        'ge': [
            f"https://www.geappliances.com/appliance/{model.upper()}",
            f"https://products.geappliances.com/appliance/gea-specs/{model.upper()}"
        ],
        'maytag': [
            f"https://www.maytag.com/support/product/{model.upper()}",
            f"https://www.maytag.com/{item_type.lower().replace(' ', '-')}/{model.lower()}"
        ]
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    brand_lower = brand.lower()
    if brand_lower in manufacturer_sites:
        for url in manufacturer_sites[brand_lower]:
            try:
                print(f"Scraping: {url}")
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract specifications
                    spec_sections = soup.find_all(['div', 'section'], class_=re.compile(r'spec|feature|detail', re.I))
                    for section in spec_sections:
                        text = section.get_text(strip=True)
                        if text and len(text) > 20 and len(text) < 500:
                            research_data['specifications'].append(text)
                    
                    # Extract features
                    feature_elements = soup.find_all(['li', 'div', 'span'], string=re.compile(r'feature|capability|function', re.I))
                    for element in feature_elements:
                        parent = element.parent
                        if parent:
                            text = parent.get_text(strip=True)
                            if text and len(text) > 10 and len(text) < 200:
                                research_data['features'].append(text)
                    
                    # Extract dimensions
                    dimension_text = soup.find_all(string=re.compile(r'dimension|size|width|height|depth|\d+\s*["\']|\d+\s*inch', re.I))
                    for dim in dimension_text:
                        if dim.strip() and len(dim.strip()) < 100:
                            research_data['dimensions'] = dim.strip()
                            break
                    
                    # Extract model numbers
                    model_numbers = re.findall(r'\b[A-Z]{2,}\d{3,}[A-Z]*\b|\b\d{3,}[A-Z]{2,}\b', response.text)
                    research_data['model_numbers'].extend(model_numbers[:5])
                    
                    # Extract part numbers
                    part_numbers = re.findall(r'part\s*#?\s*:?\s*([A-Z0-9\-]{6,})', response.text, re.I)
                    research_data['part_numbers'].extend(part_numbers[:5])
                    
                    # Extract manual links
                    manual_links = soup.find_all('a', href=re.compile(r'manual|pdf|guide', re.I))
                    for link in manual_links:
                        href = link.get('href')
                        if href:
                            if href.startswith('/'):
                                href = f"https://{brand_lower}.com{href}"
                            research_data['manual_links'].append(href)
                    
                    # Extract warranty info
                    warranty_text = soup.find_all(string=re.compile(r'warranty|guarantee|\d+\s*year', re.I))
                    for warranty in warranty_text:
                        if 'year' in warranty.lower() and len(warranty.strip()) < 100:
                            research_data['warranty'] = warranty.strip()
                            break
                
                time.sleep(1)  # Be respectful to servers
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
    
    # Additional sources: appliance databases and review sites
    additional_sources = [
        f"https://www.consumerreports.org/search/?q={brand}+{model}",
        f"https://www.homedepot.com/s/{brand}%20{model}",
        f"https://www.lowes.com/search?searchTerm={brand}+{model}",
        f"https://www.appliancepartspros.com/model-{model}.html"
    ]
    
    for url in additional_sources[:2]:  # Limit to 2 additional sources for speed
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract price information
                price_elements = soup.find_all(string=re.compile(r'\$[\d,]+', re.I))
                for price in price_elements:
                    if '$' in price and ',' in price:
                        research_data['price_range'] = price.strip()
                        break
                
                # Extract energy rating
                energy_elements = soup.find_all(string=re.compile(r'energy star|energy rating|kwh', re.I))
                for energy in energy_elements:
                    if len(energy.strip()) < 100:
                        research_data['energy_rating'] = energy.strip()
                        break
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping additional source {url}: {e}")
            continue
    
    # Clean up and deduplicate data
    research_data['specifications'] = list(set(research_data['specifications']))[:5]
    research_data['features'] = list(set(research_data['features']))[:5]
    research_data['model_numbers'] = list(set(research_data['model_numbers']))[:5]
def advanced_manufacturer_research(brand: str, model: str, item_type: str) -> dict:
    """DEPRECATED: This function has been replaced by google_based_research() for simpler, faster searches."""
    print("⚠️ advanced_manufacturer_research is deprecated, using google_based_research instead")
    return google_based_research(brand, model, item_type)


def refine_scraped_data(raw_text, item_name):
    """Clean and refine web-scraped data to extract meaningful information."""
    import re
    
    if not raw_text or len(raw_text.strip()) < 50:
        return {}
    
    refined_data = {
        'specifications': [],
        'features': [],
        'model_numbers': [],
        'part_numbers': [],
        'dimensions': '',
        'warranty': '',
        'energy_rating': '',
        'description': ''
    }
    
    # Clean the text
    cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())
    cleaned_text = re.sub(r'[^\w\s\.,\-\:\;\(\)\"/]', '', cleaned_text)
    
    # Extract model numbers (letters followed by numbers, dashes, etc.)
    model_patterns = [
        r'\b[A-Z]{2,6}[\-]?[0-9]{2,8}[A-Z]?\b',
        r'\bModel[:\s]+([A-Z0-9\-]{4,15})\b',
        r'\b[A-Z]{1,3}[0-9]{3,6}[A-Z]?\b'
    ]
    
    for pattern in model_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        refined_data['model_numbers'].extend([m.strip() for m in matches if len(m.strip()) > 3])
    
    # Extract part numbers
    part_patterns = [
        r'\bPart[:\s]+([A-Z0-9\-]{6,20})\b',
        r'\bP/N[:\s]+([A-Z0-9\-]{6,20})\b',
        r'\b[A-Z]{2,4}[0-9]{4,10}[A-Z]?\b'
    ]
    
    for pattern in part_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        refined_data['part_numbers'].extend([m.strip() for m in matches])
    
    # Extract dimensions
    dimension_patterns = [
        r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(inches?|in\.?|cm|mm)',
        r'Dimensions[:\s]+([^.]+(?:inches?|in\.?|cm|mm)[^.]*)',
        r'Size[:\s]+([^.]+(?:inches?|in\.?|cm|mm)[^.]*)'
    ]
    
    for pattern in dimension_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                refined_data['dimensions'] = ' x '.join(matches[0])
            else:
                refined_data['dimensions'] = matches[0].strip()
            break
    
    # Extract warranty information
    warranty_patterns = [
        r'Warranty[:\s]+([^.]+(?:year|month|day)[^.]*)',
        r'(\d+)[:\s]*(?:year|yr|month|mo)[s]?\s*warranty',
        r'Limited[:\s]+(\d+[^.]*(?:year|month)[^.]*)'
    ]
    
    for pattern in warranty_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            refined_data['warranty'] = matches[0].strip()
            break
    
    # Extract energy rating
    energy_patterns = [
        r'Energy[:\s]+Star[:\s]*([A-Z0-9\+\-]*)',
        r'ENERGY[:\s]+STAR[:\s]*([A-Z0-9\+\-]*)',
        r'(\d+\.?\d*)\s*kWh',
        r'Energy[:\s]+Rating[:\s]*([A-Z0-9\+\-]*)'
    ]
    
    for pattern in energy_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            refined_data['energy_rating'] = matches[0].strip()
            break
    
    # Extract features (look for bullet points or feature lists)
    feature_patterns = [
        r'(?:Features?|Benefits?|Includes?)[:\s]*([^.]+\.)',
        r'•\s*([^•\n]+)',
        r'-\s*([^-\n]+)',
        r'\*\s*([^*\n]+)'
    ]
    
    for pattern in feature_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            features = [f.strip() for f in matches if len(f.strip()) > 10 and len(f.strip()) < 200]
            refined_data['features'].extend(features[:10])  # Limit to 10 features
    
    # Extract specifications
    spec_keywords = ['capacity', 'power', 'voltage', 'frequency', 'speed', 'temperature', 'pressure']
    spec_patterns = []
    
    for keyword in spec_keywords:
        spec_patterns.append(rf'{keyword}[:\s]*([^.]+)')
    
    for pattern in spec_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            specs = [s.strip() for s in matches if len(s.strip()) > 5 and len(s.strip()) < 100]
            refined_data['specifications'].extend(specs[:5])  # Limit to 5 per keyword
    
    # Create a refined description (first substantial paragraph)
    sentences = re.split(r'[.!?]+', cleaned_text)
    description_parts = []
    
    for sentence in sentences[:5]:
        sentence = sentence.strip()
        if len(sentence) > 50 and len(sentence) < 300:
            description_parts.append(sentence)
        if len(description_parts) >= 2:
            break
    
    if description_parts:
        refined_data['description'] = '. '.join(description_parts) + '.'
    
    # Remove duplicates and clean up lists
    for key in ['specifications', 'features', 'model_numbers', 'part_numbers']:
        if refined_data[key]:
            refined_data[key] = list(set(refined_data[key]))  # Remove duplicates
            refined_data[key] = [item for item in refined_data[key] if len(item.strip()) > 3]
    
    return refined_data


def google_based_research(brand: str, model: str, item_type: str) -> dict:
    """Perform research using Google search instead of multiple manufacturer websites."""
    import re
    
    print(f"🔍 Google research: {brand} {model} {item_type}")
    
    research_data = {
        'specifications': [],
        'common_issues': [],
        'parts_info': [],
        'manual_links': [],
        'model_numbers': [],
        'part_numbers': [],
        'year_range': '',
        'price_range': '',
        'features': [],
        'dimensions': '',
        'warranty': '',
        'energy_rating': '',
        'content': ''
    }
    
    try:
        # Build comprehensive search query
        main_query = f"{brand} {model} {item_type}".strip()
        search_query = f"{main_query} specifications features manual parts repair"
        
        # Use existing search function instead of direct DuckDuckGo
        print(f"📡 Searching: {search_query}")
        search_results = search_item_online(search_query)
        
        if not search_results:
            print("⚠️ No search results found")
            return research_data
        
        print(f"🔍 Found {len(search_results)} search results")
        
        # Collect all text content from search results
        all_content = []
        for result in search_results:
            text = result.get('text', '')
            url = result.get('url', '')
            
            print(f"📄 Result: {text[:100]}...")  # Debug print
            
            # Add text content for analysis
            all_content.append(text)
            
            # Check for manual/manual links
            if any(keyword in url.lower() for keyword in ['manual', 'pdf', 'guide', 'service']):
                research_data['manual_links'].append(url)
        
        # Combine all content for analysis
        combined_content = ' '.join(all_content)
        research_data['content'] = combined_content
        
        # Use our existing refine_scraped_data function to extract information
        if len(combined_content) > 100:
            print(f"🔬 Analyzing {len(combined_content)} characters of content")
            refined_info = refine_scraped_data(combined_content, f"{brand} {model} {item_type}")
            
            # Merge refined data
            for key, value in refined_info.items():
                if isinstance(value, list) and value:
                    research_data[key] = value[:5]  # Limit to top 5 items
                    print(f"📊 Found {len(value)} {key}")
                elif isinstance(value, str) and value:
                    research_data[key] = value
                    print(f"📊 Found {key}: {value[:50]}...")
        else:
            print(f"⚠️ Not enough content to analyze: {len(combined_content)} characters")
        
        # Extract specific information from search results
        for result in search_results:
            result_text = result.get('text', '').lower()
            
            # Look for specifications
            if any(keyword in result_text for keyword in ['spec', 'specification', 'feature', 'capacity']):
                spec_text = result.get('text', '')[:200]
                if len(spec_text) > 30:
                    research_data['specifications'].append(spec_text)
            
            # Look for common issues
            if any(keyword in result_text for keyword in ['problem', 'issue', 'repair', 'fix', 'troubleshoot']):
                issue_text = result.get('text', '')[:150]
                if len(issue_text) > 20:
                    research_data['common_issues'].append(issue_text)
            
            # Look for parts information
            if any(keyword in result_text for keyword in ['part', 'replacement', 'component']):
                parts_text = result.get('text', '')[:150]
                if len(parts_text) > 20:
                    research_data['parts_info'].append(parts_text)
        
        # Remove duplicates and limit results
        for key in ['specifications', 'common_issues', 'parts_info']:
            if research_data[key]:
                research_data[key] = list(set(research_data[key]))[:5]
        
        research_data['manual_links'] = list(set(research_data['manual_links']))[:3]
        
        # Fallback: If we still have no useful data, create some basic info from the search results
        if not research_data['specifications'] and not research_data['features'] and search_results:
            print("🔧 Creating fallback specifications from search results")
            for result in search_results:
                text = result.get('text', '')
                if len(text) > 50:
                    # Use the search result text as a specification
                    research_data['specifications'].append(text[:150])
                    if len(research_data['specifications']) >= 2:
                        break
            
            # Add basic features if we have model info
            if brand and model:
                research_data['features'] = [
                    f"{brand} {model} {item_type}",
                    f"Model: {model}",
                    f"Brand: {brand}"
                ]
        
        print(f"✅ Google research complete: {len(research_data['specifications'])} specs, {len(research_data['features'])} features, {len(research_data['common_issues'])} issues")
        return research_data
        
    except Exception as e:
        print(f"❌ Google research error: {e}")
        return research_data


def comprehensive_item_research(brand: str, model: str, item_type: str) -> dict:
    """Perform comprehensive online research using simplified Google search approach."""
    print(f"🔍 Research: {brand} {model} {item_type}")
    
    # Use the new simplified Google-based research
    try:
        research_data = google_based_research(brand, model, item_type)
        print(f"✅ Research complete: {len(research_data.get('specifications', []))} specs, {len(research_data.get('features', []))} features")
        return research_data
    except Exception as e:
        print(f"❌ Research failed: {e}")
        # Return empty research data structure
        return {
            'specifications': [],
            'common_issues': [],
            'parts_info': [],
            'manual_links': [],
            'model_numbers': [],
            'part_numbers': [],
            'year_range': '',
            'price_range': '',
            'features': [],
            'dimensions': '',
            'warranty': '',
            'energy_rating': ''
        }


def suggest_item_options(user_input: str) -> list:
    """
    Analyzes user description and suggests possible item matches.
    Uses web search + AI to find real products.
{{ ... }}
    """
    # First, search the web for real products
    web_results = search_item_online(user_input)
    
    # Build enhanced prompt with web context
    web_context = ""
    if web_results:
        web_context = "\n\nWeb search results:\n"
        for i, result in enumerate(web_results[:3], 1):
            web_context += f"{i}. {result['text']}\n"
    
    prompt = f"""Based on this user description and web search results, suggest 3 REAL product matches.

User description: "{user_input}"
{web_context}

IMPORTANT: 
- Use actual brand names and model numbers from web results when available
- If web results mention specific models, prioritize those
- Extract years/age from user description
- Provide realistic confidence levels

Respond with ONLY a JSON array:
[
  {{"name": "Actual Brand Model Number", "description": "Detailed specs", "yearsUsed": X, "confidence": "high"}},
  {{"name": "Alternative Brand Model", "description": "Similar specs", "yearsUsed": X, "confidence": "medium"}},
  {{"name": "Generic Category", "description": "General type", "yearsUsed": X, "confidence": "low"}}
]

JSON Response:"""
    
    try:
        model = load_llm()
        response = model(
            prompt,
            max_tokens=768,
            temperature=0.3,  # Lower temp for more factual responses
            stop=["User:"],
            echo=False
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        # Find JSON array
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            options = json.loads(json_str)

            # Add web URLs if available
            for i, option in enumerate(options):
                if i < len(web_results):
                    option['reference_url'] = web_results[i].get('url', '')

            return options[:3]
        else:
            raise ValueError("Could not parse options from LLM response")
            
    except Exception as e:
        print(f"AI suggestion error: {e}")
        # Fallback: create options from web results
        if web_results:
            return [{
                "name": result['text'][:50],
                "description": result['text'],
                "yearsUsed": 1,
                "confidence": "medium",
                "reference_url": result.get('url', '')
            } for result in web_results[:3]]
        else:
            return [{
                "name": "Unknown Item",
                "description": user_input,
                "yearsUsed": 1,
                "confidence": "low"
            }]

def enhanced_contextual_research(item: Item, problem_context: str, conversation_history: List[dict]) -> str:
    """
    Perform enhanced research based on the specific problem context gathered from conversation.
    Returns research findings as formatted text for the AI to use.
    """
    print(f"🔬 Enhanced research for {item.name} with context: {problem_context[:50]}...")
    
    # Extract symptoms and context from conversation
    symptoms = []
    context_clues = []
    
    for msg in conversation_history:
        if msg['role'] == 'user':
            text = msg['content'].lower()
            # Look for symptom keywords
            if any(word in text for word in ['noise', 'sound', 'leak', 'not working', 'broken', 'error', 'smell', 'hot', 'cold']):
                symptoms.append(msg['content'][:100])
            # Look for context clues
            if any(word in text for word in ['when', 'after', 'before', 'during', 'always', 'sometimes']):
                context_clues.append(msg['content'][:100])
    
    # Build comprehensive research query
    research_queries = []
    
    # Base model-specific query
    base_query = f"{item.brand} {item.model} {item.category}"
    research_queries.append(f"{base_query} common problems repair parts")
    
    # Symptom-specific queries
    for symptom in symptoms[:2]:  # Top 2 symptoms
        research_queries.append(f"{base_query} {symptom} fix replacement part")
    
    # Context-specific query
    if context_clues:
        context_summary = ' '.join(context_clues[:2])
        research_queries.append(f"{base_query} {context_summary} troubleshoot")
    
    # Perform research for each query
    all_research_data = {
        'parts_found': [],
        'solutions': [],
        'part_numbers': [],
        'common_fixes': []
    }
    
    for query in research_queries[:3]:  # Limit to 3 queries for speed
        try:
            research_data = google_based_research(item.brand or '', item.model or '', query)
            
            # Extract specific information
            if research_data.get('parts_info'):
                all_research_data['parts_found'].extend(research_data['parts_info'][:2])
            
            if research_data.get('part_numbers'):
                all_research_data['part_numbers'].extend(research_data['part_numbers'][:3])
            
            if research_data.get('common_issues'):
                all_research_data['common_fixes'].extend(research_data['common_issues'][:2])
            
            if research_data.get('specifications'):
                all_research_data['solutions'].extend(research_data['specifications'][:2])
                
        except Exception as e:
            print(f"Research query failed: {e}")
            continue
    
    # Format research findings for AI consumption
    research_summary = "RESEARCH FINDINGS:\n"
    
    if all_research_data['part_numbers']:
        unique_parts = list(set(all_research_data['part_numbers']))[:5]
        research_summary += f"SPECIFIC PARTS: {', '.join(unique_parts)}\n"
    
    if all_research_data['parts_found']:
        unique_parts_info = list(set(all_research_data['parts_found']))[:3]
        research_summary += f"PARTS INFO: {'; '.join(unique_parts_info)}\n"
    
    if all_research_data['common_fixes']:
        unique_fixes = list(set(all_research_data['common_fixes']))[:3]
        research_summary += f"COMMON FIXES: {'; '.join(unique_fixes)}\n"
    
    if all_research_data['solutions']:
        unique_solutions = list(set(all_research_data['solutions']))[:2]
        research_summary += f"TECHNICAL INFO: {'; '.join(unique_solutions)}\n"
    
    print(f"✅ Research complete: {len(all_research_data['part_numbers'])} parts, {len(all_research_data['common_fixes'])} fixes")
    return research_summary


def generate_diagnosis_response(item: Item, conversation_history: List[dict]):
    """
    Generates thorough, conversational diagnosis responses that gather context,
    research extensively, and provide specific parts recommendations.
    """
    print(f"🔧 generate_diagnosis_response called for: {item.name}")
    print(f"📝 Conversation length: {len(conversation_history)}")
    
    # Build comprehensive item context for research
    item_context = f"""ITEM: {item.name}
Brand: {item.brand or 'Unknown'}
Model: {item.model or 'Unknown'}
Age: {item.yearsUsed} years
Category: {item.category or 'Unknown'}"""

    # Add technical details if available
    if hasattr(item, 'model_numbers') and item.model_numbers:
        item_context += f"\nModel Numbers: {', '.join(item.model_numbers[:2])}"
    
    if hasattr(item, 'part_numbers') and item.part_numbers:
        item_context += f"\nKnown Parts: {', '.join(item.part_numbers[:3])}"

    if item.specifications:
        item_context += f"\nSpecs: {'; '.join(item.specifications[:2])}"

    if item.common_issues:
        item_context += f"\nKnown Issues: {'; '.join(item.common_issues[:2])}"

    # Determine conversation stage and do research if needed
    conversation_length = len(conversation_history)
    research_data = ""
    
    if conversation_length <= 2:
        conversation_stage = "DISCOVERY"
    elif conversation_length <= 4:
        conversation_stage = "CONTEXT_GATHERING"
    elif conversation_length <= 6:
        conversation_stage = "RESEARCH_PHASE"
        # Perform enhanced research during this phase
        print("🔬 Entering research phase - gathering extensive model-specific data...")
        research_data = enhanced_contextual_research(item, str(conversation_history), conversation_history)
    else:
        conversation_stage = "SOLUTION_PROVIDING"
        # Also provide research data for solution phase
        if conversation_length == 7:  # First solution attempt
            research_data = enhanced_contextual_research(item, str(conversation_history), conversation_history)

    # Build stage-specific system prompt with research data
    system_context = f"""You are an expert appliance repair technician helping a customer diagnose their appliance problem.

APPLIANCE DETAILS:
{item_context}

{research_data}

YOUR ROLE:
- Ask specific diagnostic questions about symptoms
- Suggest exact replacement parts with part numbers when you identify the issue
- Be helpful and knowledgeable about this specific appliance
- Give practical repair advice

RESPONSE GUIDELINES:
- Keep responses focused and helpful (50-100 words)
- Ask ONE specific question OR provide ONE specific solution
- If suggesting parts, include exact part numbers
- Be conversational but professional

"""
    
    # Build conversation context
    full_prompt = system_context + "\nCONVERSATION:\n"
    for msg in conversation_history[-6:]:  # More context for thorough responses
        role = "Customer" if msg['role'] == 'user' else "Technician"
        content = msg['content']
        full_prompt += f"{role}: {content}\n"
    
    full_prompt += "Technician: "
    
    try:
        print(f"🤖 Loading LLM model...")
        model = load_llm()
        print(f"✅ Model loaded successfully")
        
        print(f"📝 Full prompt length: {len(full_prompt)} characters")
        print(f"🔤 Prompt preview: {full_prompt[-200:]}")  # Last 200 chars
        
        # Parameters optimized for thorough, conversational responses
        chunk_count = 0
        for output in model(
            full_prompt,
            max_tokens=300,   # More tokens for thorough explanations
            temperature=0.7,  # Higher temperature for more creative responses
            top_p=0.9,       # Allow more varied conversational language
            stop=["Customer:", "\nCustomer:"],  # Only stop on customer input
            stream=True
        ):
            chunk_count += 1
            chunk = output['choices'][0]['text']
            
            # Debug: Print what we're receiving
            print(f"🔧 AI Chunk: '{chunk}'")
            
            # Only filter completely empty chunks
            if chunk.strip():
                yield chunk
        
        print(f"🏁 Model streaming finished. Total chunks processed: {chunk_count}")
            
    except Exception as e:
        print(f"❌ Model error: {e}")
        yield f"I'm having trouble accessing my diagnostic database right now. Let me know what specific symptoms you're experiencing and I'll help troubleshoot manually."

# ==============================================================================
# --- Worker Threads for Non-Blocking Operations ---
# ==============================================================================
class DiagnosisWorker(QThread):
    """Streams diagnosis responses from the embedded LLM."""
    chunk_received = Signal(str)
    finished_streaming = Signal()
    error_occurred = Signal(str)

    def __init__(self, item: Item, conversation_history: List[dict]):
        super().__init__()
        self.item = item
        self.conversation_history = conversation_history

    def run(self):
        try:
            print(f"🤖 Starting diagnosis for: {self.item.name}")
            print(f"💬 Conversation history: {len(self.conversation_history)} messages")
            
            # Use the generator to stream responses
            chunk_count = 0
            for chunk in generate_diagnosis_response(self.item, self.conversation_history):
                chunk_count += 1
                print(f"📤 Emitting chunk #{chunk_count}: '{chunk[:50]}...'")
                self.chunk_received.emit(chunk)
            
            print(f"✅ Diagnosis complete: {chunk_count} chunks sent")
            self.finished_streaming.emit()
        except Exception as e:
            print(f"❌ Diagnosis error: {e}")
            self.error_occurred.emit(str(e))

class ItemRefinementWorker(QThread):
    """Refines user manual input into polished item descriptions."""
    refinement_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, user_data):
        super().__init__()
        self.user_data = user_data
        self._is_cancelled = False
    
    def cancel(self):
        """Cancel the current research operation."""
        self._is_cancelled = True
        self.quit()
        self.wait(2000)  # Wait up to 2 seconds for thread to finish

    def run(self):
        try:
            if self._is_cancelled:
                return
            refined_data = self.refine_item_data(self.user_data)
            if not self._is_cancelled:
                self.refinement_ready.emit(refined_data)
        except Exception as e:
            if not self._is_cancelled:
                self.error_occurred.emit(str(e))
    
    def refine_item_data(self, data):
        """Use AI and online research to create a polished, comprehensive item description."""
        if self._is_cancelled:
            return {}
            
        # First, perform online research
        brand = data['brand'] or ''
        model = data['model'] or ''
        item_type = data['type']
        
        # Quick research with timeout for speed
        research_data = {'specifications': [], 'common_issues': [], 'parts_info': [], 'manual_links': [], 'year_range': '', 'price_range': ''}
        
        try:
            if self._is_cancelled:
                return {}
            # Fast research with shorter timeout
            research_data = comprehensive_item_research(brand, model, item_type)
        except Exception as e:
            print(f"Quick research failed, using basic data: {e}")
            # Continue with empty research data for speed
        
        if self._is_cancelled:
            return {}
            
        # Build enhanced context with research data
        research_context = ""
        if research_data['specifications']:
            research_context += f"\nSpecifications found: {'; '.join(research_data['specifications'][:2])}"
        if research_data['common_issues']:
            research_context += f"\nCommon issues: {'; '.join(research_data['common_issues'][:2])}"
        if research_data['parts_info']:
            research_context += f"\nParts info: {'; '.join(research_data['parts_info'][:2])}"
        if research_data['year_range']:
            research_context += f"\nManufacture years: {research_data['year_range']}"
        if research_data['price_range']:
            research_context += f"\nPrice range: {research_data['price_range']}"
        
        prompt = f"""You are an expert at creating accurate, detailed product descriptions using real-world research data.

USER INPUT:
- Brand: {data['brand'] or 'Not specified'}
- Model: {data['model'] or 'Not specified'}
- Type: {data['type']}
- Color: {data['color'] or 'Not specified'}
- Size: {data['size'] or 'Not specified'}
- Features: {data['features'] or 'Not specified'}
- Purchase Year: {data['purchase_year'] or 'Not specified'}
- Years Used: {data['years_used'] or 'Not specified'}
- Condition: {data['condition'] or 'Not specified'}

ONLINE RESEARCH DATA:{research_context}

Extract key features from the research data and user input. Create a professional description using REAL data.

Respond with ONLY a valid JSON object:
{{
  "item_name": "Accurate Brand Model Type",
  "category": "Specific category",
  "age_years": 3,
  "purchase_year": "2020",
  "description": "Factual 1-2 sentence description using research data",
  "specifications": "Key specs from research (brief)",
  "features": "Key features extracted from research and user input",
  "condition_notes": "Current condition with any known issues",
  "estimated_value": "Based on research or 'Unknown'"
}}

JSON Response:"""
        
        if self._is_cancelled:
            return {}
            
        try:
            model = load_llm()
            if self._is_cancelled:
                return {}
            response = model(
                prompt,
                max_tokens=512,  # Increased for research context
                temperature=0.2,  # Lower for more consistent results
                top_p=0.8,       # More focused sampling
                stop=["USER INPUT:", "JSON Response:", "User:"],
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            # Find JSON object
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                refined_data = json.loads(json_str)
                
                # Add research data to the refined result
                refined_data['research_data'] = research_data
                return refined_data
            else:
                raise ValueError("Could not parse refined data from LLM response")
                
        except Exception as e:
            # Fallback: create basic refined data
            print(f"AI refinement error: {e}")
            fallback_data = self.create_fallback_refinement(data, research_data)
            fallback_data['research_data'] = research_data  # Still include research
            return fallback_data

    def create_fallback_refinement(self, data, research_data=None):
        """Create basic refinement when AI fails."""
        brand = data['brand'] or ''
        model = data['model'] or ''
        item_type = data['type']
        
        # Create name
        name_parts = [brand, model, item_type]
        item_name = ' '.join([part for part in name_parts if part])
        
        # Calculate age
        age_years = 0
        purchase_year = data.get('purchase_year')
        if purchase_year:
            try:
                current_year = 2024  # Could be made dynamic
                age_years = current_year - int(purchase_year)
            except:
                pass
        
        if not age_years and data.get('years_used'):
            try:
                age_years = int(data['years_used'])
            except:
                pass
        
        # Create description
        desc_parts = []
        if data.get('color'):
            desc_parts.append(f"{data['color']} color")
        if data.get('size'):
            desc_parts.append(data['size'])
        
        description = f"{item_type} with {' and '.join(desc_parts)}" if desc_parts else f"{item_type}"
        
        # Extract features
        features = data.get('features', '')
        if not features and research_data and research_data.get('specifications'):
            # Try to extract features from specifications
            features = '; '.join(research_data['specifications'][:2])
        
        return {
            "item_name": item_name,
            "category": item_type,
            "age_years": age_years or 1,
            "purchase_year": purchase_year or "Unknown",
            "description": description,
            "specifications": "• User-provided details\n• Manual entry",
            "features": features or "Standard features",
            "condition_notes": data.get('condition', 'Good condition'),
            "estimated_value": "To be determined"
        }

# ==============================================================================
# --- Custom Widgets ---
# ==============================================================================
class AddItemDialog(QDialog):
    """Dialog for adding a new item with two input modes."""
    item_added = Signal(Item)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Item")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(15)
        
        intro_label = QLabel("<b>Enter your item details to add them to Asclepius AI:</b>")
        intro_label.setWordWrap(True)
        self.layout.addWidget(intro_label)

        # Manual entry form
        self.manual_widget = self.create_manual_mode()
        self.layout.addWidget(self.manual_widget)

    def create_manual_mode(self):
        """Create enhanced manual input form with AI refinement."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # Instructions
        instructions = QLabel("<b>Enter your item's details and we'll refine them for accuracy:</b>")
        layout.addWidget(instructions)
        
        # Basic info section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout(basic_group)
        
        # Brand
        basic_layout.addWidget(QLabel("Brand/Manufacturer:"))
        self.manual_brand = QLineEdit()
        self.manual_brand.setPlaceholderText("e.g., Samsung, LG, Apple, Whirlpool")
        basic_layout.addWidget(self.manual_brand)
        
        # Model/Product Name
        basic_layout.addWidget(QLabel("Model/Product Name:"))
        self.manual_model = QLineEdit()
        self.manual_model.setPlaceholderText("e.g., WF45R6300AV, 55UN8800, iPhone 15 Pro")
        basic_layout.addWidget(self.manual_model)
        
        # Type/Category
        basic_layout.addWidget(QLabel("Type/Category:"))
        self.manual_type = QComboBox()
        self.manual_type.addItems([
            "Select category...",
            "Washing Machine", "Dryer", "Refrigerator", "Dishwasher", 
            "TV/Television", "Computer/Laptop", "Phone/Tablet", "Printer",
            "Microwave", "Oven/Stove", "Air Conditioner", "Heater/Furnace",
            "Car/Automobile", "Other Appliance", "Other Electronics", "Other"
        ])
        basic_layout.addWidget(self.manual_type)
        
        layout.addWidget(basic_group)
        
        # Additional details section
        details_group = QGroupBox("Additional Details (Optional)")
        details_layout = QVBoxLayout(details_group)
        
        # Color
        details_layout.addWidget(QLabel("Color:"))
        self.manual_color = QLineEdit()
        self.manual_color.setPlaceholderText("e.g., White, Black, Stainless Steel")
        details_layout.addWidget(self.manual_color)
        
        # Size/Dimensions
        details_layout.addWidget(QLabel("Size/Dimensions:"))
        self.manual_size = QLineEdit()
        self.manual_size.setPlaceholderText("e.g., 55-inch, 4.5 cu ft, Standard")
        details_layout.addWidget(self.manual_size)
        
        # Features
        details_layout.addWidget(QLabel("Key Features:"))
        self.manual_features = QTextEdit()
        self.manual_features.setPlaceholderText("e.g., Front-loading, Energy Star certified, Smart features")
        self.manual_features.setMaximumHeight(60)
        details_layout.addWidget(self.manual_features)
        
        # Purchase info
        purchase_layout = QHBoxLayout()
        
        purchase_layout.addWidget(QLabel("Purchase Year:"))
        self.manual_purchase_year = QLineEdit()
        self.manual_purchase_year.setPlaceholderText("e.g., 2020")
        self.manual_purchase_year.setMaximumWidth(80)
        purchase_layout.addWidget(self.manual_purchase_year)
        
        purchase_layout.addWidget(QLabel("Approximate Age:"))
        self.manual_years = QLineEdit()
        self.manual_years.setPlaceholderText("e.g., 3")
        self.manual_years.setMaximumWidth(60)
        purchase_layout.addWidget(self.manual_years)
        
        purchase_layout.addStretch()
        details_layout.addLayout(purchase_layout)
        
        # Current condition
        details_layout.addWidget(QLabel("Current Condition/Notes:"))
        self.manual_condition = QTextEdit()
        self.manual_condition.setPlaceholderText("Any known issues, modifications, or special notes...")
        self.manual_condition.setMaximumHeight(60)
        details_layout.addWidget(self.manual_condition)
        
        layout.addWidget(details_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.cancel_research_btn = QPushButton("Cancel Research")
        self.cancel_research_btn.setObjectName("DestructiveButton")
        self.cancel_research_btn.clicked.connect(self.cancel_research)
        self.cancel_research_btn.setVisible(False)  # Hidden initially
        button_layout.addWidget(self.cancel_research_btn)
        
        self.preview_btn = QPushButton("Preview & Refine")
        self.preview_btn.setObjectName("SecondaryButton")
        self.preview_btn.clicked.connect(self.preview_manual_item)
        button_layout.addWidget(self.preview_btn)
        
        self.manual_add_btn = QPushButton("Add Item")
        self.manual_add_btn.setObjectName("PrimaryButton")
        self.manual_add_btn.clicked.connect(self.add_manual_item)
        button_layout.addWidget(self.manual_add_btn)
        
        layout.addLayout(button_layout)
        
        # Preview area (hidden initially)
        self.preview_area = QGroupBox("AI-Refined Description")
        self.preview_layout = QVBoxLayout(self.preview_area)
        
        self.preview_text = QLabel()
        self.preview_text.setWordWrap(True)
        self.preview_text.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
        self.preview_layout.addWidget(self.preview_text)
        
        self.confirm_preview_btn = QPushButton("Use This Description")
        self.confirm_preview_btn.clicked.connect(self.confirm_preview)
        self.preview_layout.addWidget(self.confirm_preview_btn)
        
        self.preview_area.setVisible(False)
        layout.addWidget(self.preview_area)
        
        layout.addStretch()
        return widget
    
    def preview_manual_item(self):
        """Use AI to refine and create a polished description from user input."""
        # Collect all user input
        brand = self.manual_brand.text().strip()
        model = self.manual_model.text().strip()
        item_type = self.manual_type.currentText()
        color = self.manual_color.text().strip()
        size = self.manual_size.text().strip()
        features = self.manual_features.toPlainText().strip()
        purchase_year = self.manual_purchase_year.text().strip()
        years_used = self.manual_years.text().strip()
        condition = self.manual_condition.toPlainText().strip()
        
        # Basic validation
        if not brand and not model:
            QMessageBox.warning(self, "Input Required", "Please enter at least a brand or model name.")
            return
        
        if item_type == "Select category...":
            QMessageBox.warning(self, "Category Required", "Please select an item category.")
            return
        
        # Prepare data for AI refinement
        user_data = {
            "brand": brand,
            "model": model,
            "type": item_type,
            "color": color,
            "size": size,
            "features": features,
            "purchase_year": purchase_year,
            "years_used": years_used,
            "condition": condition
        }
        
        # Show loading state
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.preview_btn.setText("Researching...")
        self.preview_btn.setEnabled(False)
        self.cancel_research_btn.setVisible(True)  # Show cancel button
        self.manual_add_btn.setEnabled(False)  # Disable add button during research
        
        # Use separate thread for faster processing
        self.refine_thread = ItemRefinementWorker(user_data)
        self.refine_thread.refinement_ready.connect(self.show_refined_preview)
        self.refine_thread.error_occurred.connect(self.on_refinement_error)
        self.refine_thread.finished.connect(self.on_research_finished)  # Clean up when finished
        self.refine_thread.start()
    
    def cancel_research(self):
        """Cancel the ongoing research operation."""
        if hasattr(self, 'refine_thread') and self.refine_thread.isRunning():
            self.refine_thread.cancel()
            QApplication.restoreOverrideCursor()
            self.preview_btn.setText("Preview & Refine")
            self.preview_btn.setEnabled(True)
            self.cancel_research_btn.setVisible(False)
            self.manual_add_btn.setEnabled(True)
            print("Research cancelled by user")
    
    def on_research_finished(self):
        """Called when research thread finishes (success or failure)."""
        QApplication.restoreOverrideCursor()
        self.preview_btn.setText("Preview & Refine")
        self.preview_btn.setEnabled(True)
        self.cancel_research_btn.setVisible(False)
        self.manual_add_btn.setEnabled(True)
    
    def show_refined_preview(self, refined_data):
        """Display the AI-refined item information."""
        # Note: UI cleanup is now handled in on_research_finished
        
        # Store refined data
        self.refined_data = refined_data
        
        # Create readable description
        description = f"""<b>{refined_data['item_name']}</b>

<b>Category:</b> {refined_data['category']}
<b>Age:</b> {refined_data['age_years']} years ({refined_data.get('purchase_year', 'Unknown')} purchase year)

<b>Description:</b> {refined_data['description']}

<b>Key Features:</b>
{refined_data.get('features', 'N/A')}

<b>Key Specifications:</b>
{refined_data.get('specifications', 'N/A')}

<b>Current Condition:</b>
{refined_data.get('condition_notes', 'Good condition')}

<b>Estimated Value:</b> {refined_data.get('estimated_value', 'N/A')}
"""
        
        self.preview_text.setText(description)
        self.preview_area.setVisible(True)
        
        # Scroll to preview (if enclosed in scroll area)
        parent_widget = self.preview_area.parent()
        if isinstance(parent_widget, QWidget):
            scroll_parent = parent_widget.parent()
            if isinstance(scroll_parent, QScrollArea):
                scroll_parent.ensureWidgetVisible(self.preview_area)
    
    def confirm_preview(self):
        """Use the AI-refined data to create the item."""
        if hasattr(self, 'refined_data'):
            research = self.refined_data.get('research_data', {})
            
            new_item = Item(
                name=self.refined_data['item_name'],
                description=self.refined_data['description'],
                yearsUsed=self.refined_data['age_years'],
                # Enhanced fields from research
                brand=self.manual_brand.text().strip(),
                model=self.manual_model.text().strip(),
                category=self.refined_data.get('category', ''),
                specifications=research.get('specifications', []),
                common_issues=research.get('common_issues', []),
                parts_info=research.get('parts_info', []),
                manual_links=research.get('manual_links', []),
                year_range=research.get('year_range', ''),
                price_range=research.get('price_range', ''),
                additional_details={
                    'color': self.manual_color.text().strip(),
                    'size': self.manual_size.text().strip(),
                    'features': self.refined_data.get('features', self.manual_features.toPlainText().strip()),
                    'condition': self.manual_condition.toPlainText().strip(),
                    'purchase_year': self.manual_purchase_year.text().strip()
                }
            )
            self.item_added.emit(new_item)
            self.accept()
    
    def on_refinement_error(self, error_message):
        """Handle refinement error."""
        QApplication.restoreOverrideCursor()
        self.preview_btn.setText("Preview & Refine")
        self.preview_btn.setEnabled(True)
        self.cancel_research_btn.setVisible(False)
        self.manual_add_btn.setEnabled(True)
        QMessageBox.critical(self, "Refinement Error", f"Could not refine description:\n{error_message}")
    
    def add_manual_item(self):
        """Add item from manual input (legacy method - kept for compatibility)."""
        # Try to use refined data if available, otherwise fallback to basic input
        if hasattr(self, 'refined_data'):
            self.confirm_preview()
            return
            
        # Fallback: create basic item from direct inputs
        brand = self.manual_brand.text().strip()
        model = self.manual_model.text().strip()
        item_type = self.manual_type.currentText()
        
        if not brand and not model:
            QMessageBox.warning(self, "Input Required", "Please enter at least a brand or model name.")
            return
        
        if item_type == "Select category...":
            item_type = "Appliance"  # Default
        
        # Create basic name
        name_parts = [brand, model, item_type]
        item_name = " ".join([part for part in name_parts if part])
        
        # Create basic description
        desc_parts = []
        if self.manual_color.text().strip():
            desc_parts.append(f"Color: {self.manual_color.text().strip()}")
        if self.manual_size.text().strip():
            desc_parts.append(f"Size: {self.manual_size.text().strip()}")
        if self.manual_features.toPlainText().strip():
            desc_parts.append(f"Features: {self.manual_features.toPlainText().strip()}")
        
        description = "; ".join(desc_parts) if desc_parts else f"{item_type} entered manually"
        
        # Get age
        try:
            years = int(self.manual_years.text().strip()) if self.manual_years.text().strip() else 0
        except ValueError:
            years = 0

        new_item = Item(
            name=item_name,
            description=description,
            yearsUsed=years
        )
        self.item_added.emit(new_item)
        self.accept()

    def closeEvent(self, event):
        """Handle dialog closure - cancel any ongoing research."""
        if hasattr(self, 'refine_thread') and self.refine_thread.isRunning():
            self.refine_thread.cancel()
        event.accept()

class EditItemDialog(QDialog):
    """Dialog for editing an existing item."""
    item_updated = Signal(Item)

    def __init__(self, item: Item, parent=None):
        super().__init__(parent)
        self.current_item = item
        self.setWindowTitle(f"Edit Item: {item.name}")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(15)
        
        intro_label = QLabel(f"<b>Edit details for: {item.name}</b>")
        intro_label.setWordWrap(True)
        self.layout.addWidget(intro_label)

        # Manual entry form (similar to AddItemDialog but pre-populated)
        self.edit_widget = self.create_edit_form()
        self.layout.addWidget(self.edit_widget)

    def create_edit_form(self):
        """Create edit form pre-populated with current item data."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        instructions = QLabel("<b>Update your item's details:</b>")
        layout.addWidget(instructions)
        
        # Basic info section
        basic_group = QGroupBox("Basic Information")
        basic_layout = QVBoxLayout(basic_group)
        
        # Parse current item name and description for pre-population
        name_parts = self.current_item.name.split()
        current_description = self.current_item.description or ""
        
        # Item Name/Title
        basic_layout.addWidget(QLabel("Item Name:"))
        self.edit_name = QLineEdit()
        self.edit_name.setText(self.current_item.name)
        self.edit_name.setPlaceholderText("e.g., Samsung WF45R6300AV Washing Machine")
        basic_layout.addWidget(self.edit_name)
        
        # Description
        basic_layout.addWidget(QLabel("Description:"))
        self.edit_description = QTextEdit()
        self.edit_description.setText(current_description)
        self.edit_description.setPlaceholderText("Detailed description of the item")
        self.edit_description.setMaximumHeight(80)
        basic_layout.addWidget(self.edit_description)
        
        # Age/Years Used
        basic_layout.addWidget(QLabel("Years Used:"))
        self.edit_years = QLineEdit()
        self.edit_years.setText(str(self.current_item.yearsUsed))
        self.edit_years.setPlaceholderText("e.g., 3")
        self.edit_years.setMaximumWidth(80)
        basic_layout.addWidget(self.edit_years)
        
        layout.addWidget(basic_group)
        
        # Additional details section - Now editable
        details_group = QGroupBox("Additional Details (Editable)")
        details_layout = QVBoxLayout(details_group)
        
        # Color
        details_layout.addWidget(QLabel("Color:"))
        self.edit_color = QLineEdit()
        self.edit_color.setText(self.current_item.additional_details.get('color', ''))
        self.edit_color.setPlaceholderText("e.g., White, Black, Stainless Steel")
        details_layout.addWidget(self.edit_color)
        
        # Size
        details_layout.addWidget(QLabel("Size/Dimensions:"))
        self.edit_size = QLineEdit()
        self.edit_size.setText(self.current_item.additional_details.get('size', ''))
        self.edit_size.setPlaceholderText("e.g., 55-inch, 4.5 cu ft")
        details_layout.addWidget(self.edit_size)
        
        # Features
        details_layout.addWidget(QLabel("Features:"))
        self.edit_features = QTextEdit()
        self.edit_features.setText(self.current_item.additional_details.get('features', ''))
        self.edit_features.setPlaceholderText("Key features and capabilities")
        self.edit_features.setMaximumHeight(60)
        details_layout.addWidget(self.edit_features)
        
        # Purchase year
        details_layout.addWidget(QLabel("Purchase Year:"))
        self.edit_purchase_year = QLineEdit()
        self.edit_purchase_year.setText(self.current_item.additional_details.get('purchase_year', ''))
        self.edit_purchase_year.setPlaceholderText("e.g., 2020")
        self.edit_purchase_year.setMaximumWidth(100)
        details_layout.addWidget(self.edit_purchase_year)
        
        # Condition
        details_layout.addWidget(QLabel("Current Condition:"))
        self.edit_condition = QTextEdit()
        self.edit_condition.setText(self.current_item.additional_details.get('condition', ''))
        self.edit_condition.setPlaceholderText("Current condition and any known issues")
        self.edit_condition.setMaximumHeight(60)
        details_layout.addWidget(self.edit_condition)
        
        layout.addWidget(details_group)
        
        # Research data section (enhanced with refined data)
        if (self.current_item.specifications or self.current_item.common_issues or 
            self.current_item.features or self.current_item.model_numbers):
            research_group = QGroupBox("Research Data")
            research_layout = QVBoxLayout(research_group)
            
            # Create a scroll area for research data
            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            
            if self.current_item.specifications:
                scroll_layout.addWidget(QLabel("<b>Specifications:</b>"))
                specs_text = "\n• ".join(self.current_item.specifications[:5])
                specs_label = QLabel("• " + specs_text)
                specs_label.setWordWrap(True)
                specs_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(specs_label)
            
            if self.current_item.features:
                scroll_layout.addWidget(QLabel("<b>Features:</b>"))
                features_text = "\n• ".join(self.current_item.features[:5])
                features_label = QLabel("• " + features_text)
                features_label.setWordWrap(True)
                features_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(features_label)
            
            if self.current_item.model_numbers:
                scroll_layout.addWidget(QLabel("<b>Model Numbers:</b>"))
                models_text = ", ".join(self.current_item.model_numbers[:3])
                models_label = QLabel(models_text)
                models_label.setWordWrap(True)
                models_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(models_label)
            
            if self.current_item.part_numbers:
                scroll_layout.addWidget(QLabel("<b>Part Numbers:</b>"))
                parts_text = ", ".join(self.current_item.part_numbers[:3])
                parts_label = QLabel(parts_text)
                parts_label.setWordWrap(True)
                parts_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(parts_label)
            
            if self.current_item.dimensions:
                scroll_layout.addWidget(QLabel("<b>Dimensions:</b>"))
                dimensions_label = QLabel(self.current_item.dimensions)
                dimensions_label.setWordWrap(True)
                dimensions_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(dimensions_label)
            
            if self.current_item.warranty:
                scroll_layout.addWidget(QLabel("<b>Warranty:</b>"))
                warranty_label = QLabel(self.current_item.warranty)
                warranty_label.setWordWrap(True)
                warranty_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(warranty_label)
            
            if self.current_item.energy_rating:
                scroll_layout.addWidget(QLabel("<b>Energy Rating:</b>"))
                energy_label = QLabel(self.current_item.energy_rating)
                energy_label.setWordWrap(True)
                energy_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(energy_label)

            if self.current_item.common_issues:
                scroll_layout.addWidget(QLabel("<b>Known Issues:</b>"))
                issues_text = "\n• ".join(self.current_item.common_issues[:3])
                issues_label = QLabel("• " + issues_text)
                issues_label.setWordWrap(True)
                issues_label.setObjectName("MutedLabel")
                scroll_layout.addWidget(issues_label)
            
            scroll_area.setWidget(scroll_widget)
            scroll_area.setMaximumHeight(200)
            scroll_area.setWidgetResizable(True)
            research_layout.addWidget(scroll_area)
            
            layout.addWidget(research_group)
        
        # Diagnosis history section
        history_group = QGroupBox("Diagnosis History")
        history_layout = QVBoxLayout(history_group)
        
        # Show diagnosis history count
        history_count = len(self.current_item.diagnosisHistory)
        history_label = QLabel(f"Diagnosis History: {history_count} entries")
        history_label.setObjectName("MutedLabel")
        details_layout.addWidget(history_label)
        
        if history_count > 0:
            # Show latest diagnosis
            latest_diagnosis = self.current_item.diagnosisHistory[-1]
            latest_label = QLabel(f"Latest: {latest_diagnosis.date} - {latest_diagnosis.issuesReported[:50]}...")
            latest_label.setObjectName("MutedLabel")
            latest_label.setWordWrap(True)
            details_layout.addWidget(latest_label)
        
        layout.addWidget(details_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.clicked.connect(self.save_changes)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("SecondaryButton")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        return widget
    
    def save_changes(self):
        """Save the changes to the item."""
        # Validate input
        new_name = self.edit_name.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Input", "Item name cannot be empty.")
            return
        
        try:
            new_years = int(self.edit_years.text().strip())
            if new_years < 0:
                raise ValueError("Years cannot be negative")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Years used must be a valid positive number.")
            return
        
        # Update the item
        self.current_item.name = new_name
        self.current_item.description = self.edit_description.toPlainText().strip()
        self.current_item.yearsUsed = new_years
        
        # Update additional details
        self.current_item.additional_details.update({
            'color': self.edit_color.text().strip(),
            'size': self.edit_size.text().strip(),
            'features': self.edit_features.toPlainText().strip(),
            'purchase_year': self.edit_purchase_year.text().strip(),
            'condition': self.edit_condition.toPlainText().strip()
        })
        
        # Emit signal with updated item
        self.item_updated.emit(self.current_item)
        self.accept()


class ItemCard(QFrame):
    """A custom widget to display a single item's information."""
    diagnosing_requested = Signal(str)
    edit_requested = Signal(str)
    delete_requested = Signal(str)
    def __init__(self, item: Item, parent=None):
        super().__init__(parent)
        self.item_id = item.id
        self.setObjectName("ItemCard")
        
        layout = QVBoxLayout(self)

        # Basic info section
        name_label = QLabel(f"<b>{item.name or 'Unnamed Item'}</b>")
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

        desc_label = QLabel(item.description or "No description provided")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setObjectName("MutedLabel")
        layout.addWidget(desc_label)

        age_label = QLabel(f"Age: {item.yearsUsed} year(s)")
        age_label.setAlignment(Qt.AlignCenter)
        age_label.setObjectName("MutedLabel")
        layout.addWidget(age_label)

        layout.addSpacing(6)

        # Buttons section
        button_layout = QHBoxLayout()
        
        self.diagnose_button = QPushButton("🔍 Diagnose")
        self.diagnose_button.setObjectName("PrimaryButton")
        self.diagnose_button.clicked.connect(lambda: self.diagnosing_requested.emit(self.item_id))
        
        self.edit_button = QPushButton("✏️ Edit")
        self.edit_button.setObjectName("SecondaryButton")
        self.edit_button.clicked.connect(lambda: self.edit_requested.emit(self.item_id))
        
        self.delete_button = QPushButton("🗑️ Delete")
        self.delete_button.setObjectName("DangerButton")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.item_id))
        
        button_layout.addWidget(self.diagnose_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.delete_button)
        
        layout.addLayout(button_layout)


class ChatMessage(QFrame):
    """A single chat message bubble."""
    def __init__(self, content: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.setObjectName("UserMessage" if is_user else "AssistantMessage")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(content)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        layout.addWidget(self.label)
        
    def append_text(self, text: str):
        """Append text to the message (for streaming)."""
        current = self.label.text()
        self.label.setText(current + text)


class DiagnosisPage(QWidget):
    """Interactive diagnosis page with LLM chat interface."""
    back_requested = Signal()
    diagnosis_completed = Signal(str, list)  # item_id, suggested_parts
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.conversation_history = []
        self.diagnosis_worker = None
        self.current_assistant_message = None
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        self.back_button = QPushButton("← Back to Dashboard")
        self.back_button.clicked.connect(self.back_requested.emit)
        
        self.title_label = QLabel("Diagnosis")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        header_layout.addWidget(self.back_button)
        header_layout.addStretch()
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        
        # Item details
        self.item_details_label = QLabel()
        self.item_details_label.setWordWrap(True)
        self.item_details_label.setObjectName("MutedLabel")
        
        # Chat area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(12)
        
        self.chat_scroll.setWidget(self.chat_container)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Describe the problem or answer the assistant's question...")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("PrimaryButton")
        self.send_button.clicked.connect(self.send_message)
        
        self.find_parts_button = QPushButton("🔍 Find Parts Online")
        self.find_parts_button.clicked.connect(self.find_parts_online)
        self.find_parts_button.setEnabled(False)
        
        self.find_technician_button = QPushButton("👨‍🔧 Find Technician")
        self.find_technician_button.clicked.connect(self.find_technician_online)
        self.find_technician_button.setEnabled(False)
        
        self.save_button = QPushButton("Save Diagnosis")
        self.save_button.clicked.connect(self.save_diagnosis)
        self.save_button.setEnabled(False)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.find_parts_button)
        input_layout.addWidget(self.find_technician_button)
        input_layout.addWidget(self.save_button)
        
        # Add all to main layout
        self.layout.addLayout(header_layout)
        self.layout.addWidget(self.item_details_label)
        self.layout.addWidget(self.chat_scroll, 1)
        self.layout.addLayout(input_layout)

    def set_item(self, item: Item):
        """Initialize diagnosis for a new item."""
        self.current_item = item
        self.conversation_history = []
        self.title_label.setText(f"Diagnosing: {item.name}")
        self.item_details_label.setText(
            f"<b>Description:</b> {item.description} | <b>Age:</b> {item.yearsUsed} year(s)"
        )
        
        # Clear chat
        while self.chat_layout.count():
            child = self.chat_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add initial assistant message focused on DIY repair
        initial_msg = (
            f"I'm here to help you fix your {item.name} yourself! "
            f"Tell me exactly what's wrong - what symptoms are you seeing? "
            f"I'll identify the parts you need and guide you through the repair."
        )
        self.add_message(initial_msg, is_user=False)
        self.conversation_history.append({"role": "assistant", "content": initial_msg})
        
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.save_button.setEnabled(False)

    def add_message(self, content: str, is_user: bool):
        """Add a message to the chat."""
        message = ChatMessage(content, is_user)
        
        # Align messages
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(message, 0, Qt.AlignRight)
        else:
            container_layout.addWidget(message, 0, Qt.AlignLeft)
            container_layout.addStretch()
        
        self.chat_layout.addWidget(container)
        
        # Auto-scroll to bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
        
        return message

    def scroll_to_bottom(self):
        """Scroll chat to bottom."""
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def send_message(self):
        """Send user message and get LLM response."""
        user_input = self.input_field.text().strip()
        if not user_input or not self.current_item:
            return
        
        # Add user message
        self.add_message(user_input, is_user=True)
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Clear input and disable
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Start streaming assistant response
        self.current_assistant_message = self.add_message("", is_user=False)
        
        self.diagnosis_worker = DiagnosisWorker(self.current_item, self.conversation_history)
        self.diagnosis_worker.chunk_received.connect(self.on_chunk_received)
        self.diagnosis_worker.finished_streaming.connect(self.on_streaming_finished)
        self.diagnosis_worker.error_occurred.connect(self.on_diagnosis_error)
        self.diagnosis_worker.start()

    @Slot(str)
    def on_chunk_received(self, chunk: str):
        """Handle streaming chunk from LLM."""
        if self.current_assistant_message:
            self.current_assistant_message.append_text(chunk)
            self.scroll_to_bottom()

    @Slot()
    def on_streaming_finished(self):
        """Handle completion of streaming response."""
        if self.current_assistant_message:
            assistant_text = self.current_assistant_message.label.text()
            self.conversation_history.append({"role": "assistant", "content": assistant_text})
        
        self.current_assistant_message = None
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.find_parts_button.setEnabled(True)  # Enable parts search after diagnosis
        self.find_technician_button.setEnabled(True)  # Enable technician search after diagnosis

    @Slot(str)
    def on_diagnosis_error(self, error_msg: str):
        """Handle diagnosis error."""
        QMessageBox.critical(self, "Diagnosis Error", f"Failed to get response from LLM:\n{error_msg}")
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

    def extract_parts_from_diagnosis(self) -> list:
        """Use AI to extract specific part names with model/part numbers from diagnosis conversation."""
        if not self.conversation_history:
            return []
        
        # Build conversation summary focusing on parts mentioned
        conversation_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.conversation_history[-6:]  # Last 6 messages for context
        ])
        
        # Build enhanced context with item details
        item_details = f"""
Item: {self.current_item.name}
Brand: {self.current_item.brand or 'Unknown'}
Model: {self.current_item.model or 'Unknown'}
Category: {self.current_item.category or 'Unknown'}"""

        # Add model numbers if available
        if hasattr(self.current_item, 'model_numbers') and self.current_item.model_numbers:
            item_details += f"\nModel Numbers: {', '.join(self.current_item.model_numbers[:3])}"

        # Add known part numbers if available
        if hasattr(self.current_item, 'part_numbers') and self.current_item.part_numbers:
            item_details += f"\nKnown Part Numbers: {', '.join(self.current_item.part_numbers[:3])}"

        prompt = f"""Extract SPECIFIC replacement parts from this repair conversation. Focus on parts that need to be purchased for DIY repair.

ITEM CONTEXT:{item_details}

CONVERSATION:
{conversation_text[:1000]}

Extract parts mentioned or implied. Include:
1. Exact part names (e.g., "door latch assembly", "heating element")
2. Model/part numbers if mentioned or can be inferred from item model
3. OEM vs aftermarket compatibility

Respond with ONLY a JSON array:
[
  {{
    "part_name": "Specific Part Name",
    "part_number": "Model/Part Number or null",
    "description": "Why this part is needed",
    "oem_compatible": "Brand name if OEM available",
    "search_terms": "Best search terms for online shopping"
  }}
]

JSON Response:"""
        
        try:
            model = load_llm()
            response = model(
                prompt,
                max_tokens=400,  # Increased for more detailed parts info
                temperature=0.2,  # Lower for more accurate part extraction
                top_p=0.7,
                stop=["Conversation:", "User:", "ITEM CONTEXT:"],
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            
            # Find JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parts = json.loads(json_str)
                
                # Enhance parts with item context if missing
                for part in parts:
                    if not part.get('part_number') or part['part_number'] == 'null':
                        # Try to construct part number from item model
                        if self.current_item.brand and self.current_item.model:
                            part['search_terms'] = f"{self.current_item.brand} {self.current_item.model} {part['part_name']}"
                    
                    # Add OEM info if brand is known
                    if not part.get('oem_compatible') and self.current_item.brand:
                        part['oem_compatible'] = self.current_item.brand
                
                return parts[:5]  # Limit to 5 parts for practicality
            
        except Exception as e:
            print(f"Part extraction error: {e}")
        
        # Fallback: create basic parts from conversation keywords
        parts_keywords = ['part', 'replace', 'component', 'element', 'assembly', 'motor', 'belt', 'filter', 'valve', 'switch']
        conversation_lower = conversation_text.lower()
        
        fallback_parts = []
        for keyword in parts_keywords:
            if keyword in conversation_lower:
                fallback_parts.append({
                    "part_name": f"{keyword.title()} (from conversation)",
                    "part_number": None,
                    "description": "Mentioned in repair discussion",
                    "oem_compatible": self.current_item.brand or "Universal",
                    "search_terms": f"{self.current_item.name} {keyword}"
                })
        
        return fallback_parts[:3] if fallback_parts else []
    
    def extract_repair_context(self) -> dict:
        """Extract repair type and context from the diagnosis conversation."""
        if not self.conversation_history:
            return {
                'item_type': self.current_item.name.split()[0],
                'repair_type': 'repair',
                'search_queries': [
                    f"{self.current_item.name} repair technician",
                    f"appliance repair service",
                    f"professional repair technician"
                ]
            }
        
        # Build conversation text
        conversation_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.conversation_history
        ])
        
        # Extract key information
        item_name = self.current_item.name
        item_type = item_name.split()[0]  # Brand or category
        
        # Analyze conversation for repair type
        repair_keywords = {
            'appliance': ['washing machine', 'dryer', 'refrigerator', 'dishwasher', 'oven', 'stove', 'microwave'],
            'electronics': ['tv', 'television', 'computer', 'laptop', 'monitor', 'phone', 'tablet'],
            'automotive': ['car', 'engine', 'transmission', 'brakes', 'battery'],
            'hvac': ['heating', 'cooling', 'furnace', 'air conditioner', 'ductwork'],
            'plumbing': ['pipe', 'faucet', 'toilet', 'water heater', 'leak'],
            'electrical': ['wiring', 'outlet', 'switch', 'circuit', 'breaker']
        }
        
        # Determine repair category
        repair_category = 'general'
        conversation_lower = conversation_text.lower()
        
        for category, keywords in repair_keywords.items():
            if any(keyword in conversation_lower for keyword in keywords):
                repair_category = category
                break
        
        # Create contextual search queries
        if repair_category == 'appliance':
            search_queries = [
                f"{item_name} appliance repair technician",
                f"{repair_category} repair service",
                f"certified appliance technician",
                f"home appliance repair specialist"
            ]
        elif repair_category == 'electronics':
            search_queries = [
                f"{item_name} electronics repair technician",
                f"{repair_category} repair service",
                f"certified electronics technician",
                f"device repair specialist"
            ]
        elif repair_category == 'automotive':
            search_queries = [
                f"{item_name} auto repair technician",
                f"automotive repair service",
                f"certified mechanic",
                f"auto repair shop"
            ]
        elif repair_category in ['hvac', 'plumbing', 'electrical']:
            search_queries = [
                f"{repair_category} repair technician",
                f"{repair_category} service professional",
                f"licensed {repair_category} technician",
                f"certified {repair_category} specialist"
            ]
        else:
            search_queries = [
                f"{item_name} repair technician",
                f"professional repair service",
                f"certified repair technician",
                f"home repair specialist"
            ]
        
        return {
            'item_name': item_name,
            'item_type': item_type,
            'repair_category': repair_category,
            'search_queries': search_queries,
            'conversation_summary': conversation_text[:200]  # First 200 chars for context
        }
    
    def find_parts_online(self):
        """Search for replacement parts online based on diagnosed issues."""
        if not self.current_item:
            return
        
        # Extract specific parts from diagnosis
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        extracted_parts = self.extract_parts_from_diagnosis()
        QApplication.restoreOverrideCursor()
        
        if not extracted_parts:
            QMessageBox.information(
                self, 
                "No Parts Found", 
                "No specific parts were mentioned in the diagnosis yet.\n\n"
                "Continue the conversation to identify which parts need replacement."
            )
            return
        
        # Show parts selection dialog
        self.show_parts_selection(extracted_parts)
        # Note: Both buttons stay enabled
    
    def find_technician_online(self):
        """Search for local technicians based on diagnosed issues."""
        if not self.current_item:
            return
        
        # Show technician location dialog
        self.show_technician_location_dialog()
    
    def show_technician_location_dialog(self):
        """Show dialog to collect location for technician search."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Find Local Technician")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"<b>Find Technician for: {self.current_item.name}</b>"))
        layout.addWidget(QLabel("Please provide your location to find nearby technicians:"))
        
        # Country
        layout.addWidget(QLabel("Country:"))
        self.country_edit = QLineEdit()
        self.country_edit.setPlaceholderText("e.g., United States")
        layout.addWidget(self.country_edit)
        
        # State/Province
        layout.addWidget(QLabel("State/Province:"))
        self.state_edit = QLineEdit()
        self.state_edit.setPlaceholderText("e.g., California")
        layout.addWidget(self.state_edit)
        
        # Zip/Postal Code
        layout.addWidget(QLabel("Zip/Postal Code:"))
        self.zip_edit = QLineEdit()
        self.zip_edit.setPlaceholderText("e.g., 90210")
        layout.addWidget(self.zip_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        search_btn = QPushButton("🔍 Search for Technicians")
        search_btn.setObjectName("PrimaryButton")
        search_btn.clicked.connect(lambda: self.search_technicians(dialog))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(search_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def search_technicians(self, parent_dialog):
        """Search for technicians based on location and diagnosis context."""
        country = self.country_edit.text().strip()
        state = self.state_edit.text().strip()
        zip_code = self.zip_edit.text().strip()
        
        if not country or not state or not zip_code:
            QMessageBox.warning(self, "Location Required", "Please fill in all location fields.")
            return
        
        # Close location dialog
        parent_dialog.accept()
        
        # Extract repair type from conversation
        repair_context = self.extract_repair_context()
        
        # Build search queries based on context
        location = f"{zip_code}, {state}, {country}"
        
        # Show results dialog with contextual searches
        self.show_technician_results(repair_context, location)
    
    def show_technician_results(self, repair_context, location):
        """Show local small business technician search results."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Local Repair Specialists")
        dialog.setMinimumWidth(650)
        dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout(dialog)
        
        # Enhanced header with context
        item_name = repair_context.get('item_name', self.current_item.name)
        repair_category = repair_context.get('repair_category', 'general')
        
        layout.addWidget(QLabel(f"<h3>👨‍🔧 Local Specialists near {location}</h3>"))
        layout.addWidget(QLabel(f"For: {item_name} ({repair_category.title()} Repair)"))
        
        if repair_context.get('conversation_summary'):
            summary = repair_context['conversation_summary']
            if len(summary) > 150:
                summary = summary[:147] + "..."
            layout.addWidget(QLabel(f"<i>Issue: {summary}</i>"))
        
        # LOCAL BUSINESS FOCUSED DIRECTORIES
        layout.addWidget(QLabel("<b>🏪 Local Small Business Directories:</b>"))
        
        search_queries = repair_context.get('search_queries', [f"{item_name} repair"])
        location_encoded = '+'.join(location.split())
        service_encoded = '+'.join(search_queries[0].split())
        
        # Focus on local business directories
        local_directories = [
            {
                "name": "Google Maps Local", 
                "url": f"https://www.google.com/maps/search/{service_encoded}+near+{location_encoded}",
                "icon": "🗺️",
                "description": "Find nearby shops with reviews"
            },
            {
                "name": "Yelp Local Business", 
                "url": f"https://www.yelp.com/search?find_desc={service_encoded}&find_loc={location_encoded}&start=0&sortby=rating",
                "icon": "⭐",
                "description": "Highly rated local repair shops"
            },
            {
                "name": "NextDoor Recommendations", 
                "url": f"https://nextdoor.com/search/?query={service_encoded}",
                "icon": "🏘️",
                "description": "Neighbor-recommended technicians"
            },
            {
                "name": "Local Facebook Groups", 
                "url": f"https://www.facebook.com/search/posts/?q={service_encoded}+{location_encoded}+repair+recommendation",
                "icon": "👥",
                "description": "Community recommendations"
            },
            {
                "name": "Better Business Bureau", 
                "url": f"https://www.bbb.org/search?find_country=USA&find_text={service_encoded}&find_loc={location_encoded}",
                "icon": "🛡️",
                "description": "BBB accredited local businesses"
            }
        ]
        
        for directory in local_directories:
            btn_layout = QHBoxLayout()
            
            btn = QPushButton(f"{directory['icon']} {directory['name']}")
            btn.setMinimumHeight(35)
            btn.clicked.connect(lambda checked, url=directory['url']: self.open_url(url))
            
            desc_label = QLabel(f"- {directory['description']}")
            desc_label.setObjectName("MutedLabel")
            desc_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            btn_layout.addWidget(btn, 1)
            btn_layout.addWidget(desc_label, 2)
            
            layout.addLayout(btn_layout)
        
        # BRAND-SPECIFIC AUTHORIZED REPAIR (if applicable)
        if self.current_item.brand and self.current_item.brand.lower() in ['samsung', 'lg', 'whirlpool', 'ge', 'maytag', 'kitchenaid']:
            layout.addWidget(QLabel("<b>🏭 Authorized Service Centers:</b>"))
            brand = self.current_item.brand
            
            authorized_searches = [
                {
                    "name": f"{brand} Authorized Service",
                    "url": f"https://www.google.com/search?q={brand}+authorized+service+center+{location_encoded}",
                    "icon": "⚙️"
                },
                {
                    "name": f"{brand} Parts & Service",
                    "url": f"https://www.google.com/search?q={brand}+parts+service+repair+{location_encoded}",
                    "icon": "🔧"
                }
            ]
            
            for auth in authorized_searches:
                btn = QPushButton(f"{auth['icon']} {auth['name']}")
                btn.clicked.connect(lambda checked, url=auth['url']: self.open_url(url))
                layout.addWidget(btn)
        
        # TIPS FOR CHOOSING LOCAL REPAIR SHOPS
        layout.addWidget(QLabel("<b>💡 Tips for Choosing Local Repair Shops:</b>"))
        
        tips = [
            "✓ Check Google/Yelp reviews (look for 4+ stars)",
            "✓ Ask neighbors for recommendations on NextDoor",
            "✓ Verify business license and insurance",
            "✓ Get written quotes before work begins",
            "✓ Ask about warranties on repair work",
            "✓ Choose shops that specialize in your item type",
            "✓ Avoid unusually low prices (quality concerns)",
            "✓ Check Better Business Bureau ratings"
        ]
        
        tips_widget = QWidget()
        tips_layout = QVBoxLayout(tips_widget)
        
        for tip in tips:
            tip_label = QLabel(tip)
            tip_label.setWordWrap(True)
            tips_layout.addWidget(tip_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tips_widget)
        scroll.setMaximumHeight(120)
        layout.addWidget(scroll)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_parts_selection(self, parts):
        """Show extracted parts with enhanced information for DIY repair."""
        dialog = QDialog(self)
        dialog.setWindowTitle("DIY Repair Parts Needed")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"<b>🔧 Parts needed for: {self.current_item.name}</b>"))
        layout.addWidget(QLabel("Select a part to find and purchase online:"))
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        parts_widget = QWidget()
        parts_layout = QVBoxLayout(parts_widget)
        
        for part in parts:
            part_frame = QFrame()
            part_frame.setObjectName("ItemCard")
            part_frame_layout = QVBoxLayout(part_frame)
            
            # Part name
            part_name = QLabel(f"<b>{part['part_name']}</b>")
            part_frame_layout.addWidget(part_name)
            
            # Description
            part_desc = QLabel(part.get('description', 'No description'))
            part_desc.setWordWrap(True)
            part_desc.setObjectName("MutedLabel")
            part_frame_layout.addWidget(part_desc)
            
            # Part number
            if part.get('part_number') and part['part_number'] != 'null':
                part_num = QLabel(f"Part #: {part['part_number']}")
                part_num.setObjectName("MutedLabel")
                part_frame_layout.addWidget(part_num)
            
            # OEM compatibility
            if part.get('oem_compatible'):
                oem_label = QLabel(f"OEM Compatible: {part['oem_compatible']}")
                oem_label.setObjectName("MutedLabel")
                part_frame_layout.addWidget(oem_label)
            
            # Search terms
            if part.get('search_terms'):
                search_label = QLabel(f"Search: {part['search_terms']}")
                search_label.setObjectName("MutedLabel")
                search_label.setWordWrap(True)
                part_frame_layout.addWidget(search_label)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            search_btn = QPushButton("🔍 Find This Part")
            search_btn.setObjectName("PrimaryButton")
            search_btn.clicked.connect(
                lambda checked, p=part: self.search_specific_part(p, dialog)
            )
            
            info_btn = QPushButton("ℹ️ Part Info")
            info_btn.setObjectName("SecondaryButton")
            info_btn.clicked.connect(
                lambda checked, p=part: self.show_part_info(p)
            )
            
            button_layout.addWidget(search_btn)
            button_layout.addWidget(info_btn)
            part_frame_layout.addLayout(button_layout)
            
            parts_layout.addWidget(part_frame)
        
        scroll.setWidget(parts_widget)
        layout.addWidget(scroll)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_part_info(self, part):
        """Show detailed information about a specific part."""
        info_dialog = QDialog(self)
        info_dialog.setWindowTitle(f"Part Info: {part['part_name']}")
        info_dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(info_dialog)
        
        info_text = f"""<h3>{part['part_name']}</h3>
        
<b>Description:</b> {part.get('description', 'N/A')}

<b>Part Number:</b> {part.get('part_number', 'Generic/Universal')}

<b>OEM Compatibility:</b> {part.get('oem_compatible', 'Unknown')}

<b>Search Terms:</b> {part.get('search_terms', 'N/A')}

<b>Repair Tip:</b> When ordering, verify compatibility with your exact model number: {self.current_item.model or 'Check your item label'}
"""
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(info_dialog.accept)
        layout.addWidget(close_btn)
        
        info_dialog.exec()
    
    def search_specific_part(self, part, parent_dialog):
        """Search for a specific part online with comprehensive context."""
        from parts_lookup import search_parts_comprehensive
        import webbrowser
        
        # Build detailed search query using ALL available item context
        item = self.current_item
        part_name = part['part_name']
        part_number = part.get('part_number', '')
        
        # Create comprehensive search context
        search_context = {
            'item_name': item.name,
            'brand': item.brand or '',
            'model': item.model or '',
            'category': item.category or '',
            'part_name': part_name,
            'part_number': part_number,
            'year_range': item.year_range or '',
            'specifications': item.specifications[:2] if item.specifications else [],
            'additional_details': item.additional_details
        }
        
        # Build intelligent search query
        base_query_parts = []
        
        # Start with brand and model for specificity
        if search_context['brand']:
            base_query_parts.append(search_context['brand'])
        if search_context['model']:
            base_query_parts.append(search_context['model'])
        
        # Add part name
        base_query_parts.append(part_name)
        
        # Add part number if available
        if part_number and part_number != "null":
            base_query_parts.append(part_number)
        
        # Add year for compatibility
        if search_context['year_range']:
            year_match = search_context['year_range'].split('-')[0]  # Take first year
            base_query_parts.append(year_match)
        
        # Create primary and alternative search queries
        primary_query = " ".join(base_query_parts)
        alternative_query = f"{search_context['category']} {part_name} replacement"
        
        try:
            # Get comprehensive results with enhanced context
            results = search_parts_comprehensive(search_context['item_name'], part_name)
            
            # Auto-open with intelligent search
            from urllib.parse import quote_plus
            
            # Try specific search first
            amazon_search = quote_plus(primary_query)
            amazon_url = f"https://www.amazon.com/s?k={amazon_search}&rh=n%3A15684181"  # Automotive parts category
            
            # For appliances, use different category
            if 'appliance' in search_context['category'].lower() or any(word in item.name.lower() for word in ['washer', 'dryer', 'refrigerator', 'dishwasher']):
                amazon_url = f"https://www.amazon.com/s?k={amazon_search}&rh=n%3A2619525011"  # Appliance parts
            
            webbrowser.open(amazon_url)
            
            parent_dialog.accept()  # Close selection dialog
            self.show_enhanced_parts_results(results, part_name, primary_query, search_context)
            
        except Exception as e:
            QMessageBox.warning(self, "Search Error", f"Could not search for parts:\n{str(e)}")
    
    def show_enhanced_parts_results(self, results, part_name, search_query, search_context):
        """Display enhanced parts search results with context-aware links."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Parts for: {search_context['item_name']}")
        dialog.setMinimumWidth(700)
        dialog.setMinimumHeight(450)
        
        layout = QVBoxLayout(dialog)
        
        # Enhanced header with context
        header = QLabel(f"<h3>� Finding: {part_name}</h3>")
        layout.addWidget(header)
        
        # Context display
        context_text = f"<b>Item:</b> {search_context['brand']} {search_context['model']}"
        if search_context['year_range']:
            context_text += f" ({search_context['year_range']})"
        context_label = QLabel(context_text)
        context_label.setObjectName("MutedLabel")
        layout.addWidget(context_label)
        
        query_label = QLabel(f"<i>Search: {search_query}</i>")
        query_label.setObjectName("MutedLabel")
        layout.addWidget(query_label)
        
        # Enhanced shopping links with better targeting
        layout.addWidget(QLabel("<b>🛒 Targeted Parts Sources:</b>"))
        
        from urllib.parse import quote_plus
        encoded_query = quote_plus(search_query)
        encoded_alt = quote_plus(f"{search_context['category']} {part_name}")
        
        # Context-aware shopping sites
        shopping_sites = [
            {
                "name": "Amazon Parts", 
                "url": f"https://www.amazon.com/s?k={encoded_query}&rh=n%3A15684181",
                "icon": "🛒"
            },
            {
                "name": "eBay Parts", 
                "url": f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}&_sacat=6028",
                "icon": "🏪"
            },
            {
                "name": "Parts Direct", 
                "url": f"https://www.google.com/search?q={encoded_query}+parts+site%3A*.com",
                "icon": "🔍"
            }
        ]
        
        # Add brand-specific sources if brand is known
        if search_context['brand'].lower() in ['samsung', 'lg', 'whirlpool', 'ge']:
            brand_search = quote_plus(f"{search_context['brand']} {part_name} genuine OEM")
            shopping_sites.insert(1, {
                "name": f"{search_context['brand']} OEM Parts",
                "url": f"https://www.google.com/search?q={brand_search}",
                "icon": "⚙️"
            })
        
        for site in shopping_sites:
            btn = QPushButton(f"{site['icon']} {site['name']}")
            btn.clicked.connect(lambda checked, url=site['url']: self.open_url(url))
            layout.addWidget(btn)
        
        # Enhanced repair guides section
        if results.get('guides'):
            layout.addWidget(QLabel("<b>📚 Repair Guides:</b>"))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            guides_widget = QWidget()
            guides_layout = QVBoxLayout(guides_widget)
            
            for guide in results['guides'][:3]:  # Limit to 3 guides
                guide_frame = QFrame()
                guide_frame.setObjectName("ItemCard")
                guide_frame_layout = QVBoxLayout(guide_frame)
                
                title_label = QLabel(f"<b>{guide.get('title', 'Repair Guide')}</b>")
                title_label.setWordWrap(True)
                guide_frame_layout.addWidget(title_label)
                
                if guide.get('description'):
                    desc_label = QLabel(guide['description'][:100] + "...")
                    desc_label.setWordWrap(True)
                    desc_label.setObjectName("MutedLabel")
                    guide_frame_layout.addWidget(desc_label)
                
                if guide.get('url'):
                    view_btn = QPushButton("View Guide")
                    view_btn.clicked.connect(lambda checked, url=guide['url']: self.open_url(url))
                    guide_frame_layout.addWidget(view_btn)
                
                guides_layout.addWidget(guide_frame)
            
            scroll.setWidget(guides_widget)
            scroll.setMaximumHeight(200)
            layout.addWidget(scroll)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def open_url(self, url):
        """Open URL in default browser."""
        import webbrowser
        webbrowser.open(url)

    def save_diagnosis(self):
        """Save the diagnosis session to item history."""
        if not self.current_item or len(self.conversation_history) < 2:
            return
        
        # Extract suggested parts from conversation (simple keyword search)
        suggested_parts = []
        full_conversation = "\n".join([msg['content'] for msg in self.conversation_history])
        
        # Look for common part-related keywords
        if "part" in full_conversation.lower() or "replace" in full_conversation.lower():
            suggested_parts.append("See conversation for details")
        
        self.diagnosis_completed.emit(self.current_item.id, suggested_parts)
        QMessageBox.information(self, "Saved", "Diagnosis has been saved to item history.")
        self.save_button.setEnabled(False)

# ==============================================================================
# --- Main Application Window ---
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Asclepius AI v{APP_VERSION}")
        self.setMinimumSize(800, 600)
        self.items: List[Item] = []

        # --- Main View Management ---
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # --- Create Pages ---
        self.dashboard_widget = QWidget()
        self.diagnosis_page = DiagnosisPage()
        
        self.stacked_widget.addWidget(self.dashboard_widget)
        self.stacked_widget.addWidget(self.diagnosis_page)
        
        self.setup_dashboard_ui()
        
        # --- Connect Signals ---
        self.diagnosis_page.back_requested.connect(self.show_dashboard)
        self.diagnosis_page.diagnosis_completed.connect(self.save_diagnosis_to_item)
        
        self.load_items()
        self.refresh_dashboard()

    def setup_dashboard_ui(self):
        layout = QVBoxLayout(self.dashboard_widget)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("<b>Your Items</b>")
        title_label.setStyleSheet("font-size: 20px;")
        self.add_item_button = QPushButton("Add New Item")
        self.add_item_button.setIconSize(QSize(16, 16))
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.add_item_button)
        
        # Item Grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("ScrollArea")
        
        self.item_grid_container = QWidget()
        self.item_grid_layout = QGridLayout(self.item_grid_container)
        self.item_grid_layout.setAlignment(Qt.AlignTop)
        
        scroll_area.setWidget(self.item_grid_container)
        
        layout.addLayout(header_layout)
        layout.addWidget(scroll_area)
        
        self.add_item_button.clicked.connect(self.open_add_item_dialog)
        
    def refresh_dashboard(self):
        # Clear existing widgets from the grid
        while self.item_grid_layout.count():
            child = self.item_grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.items:
            # Show empty state message
            empty_label = QLabel("<h2>No items yet</h2><p>Click 'Add New Item' to get started.</p>")
            empty_label.setAlignment(Qt.AlignCenter)
            self.item_grid_layout.addWidget(empty_label, 0, 0, 1, 3)
            return

        # Populate grid with item cards
        row, col = 0, 0
        for item in self.items:
            card = ItemCard(item)
            card.diagnosing_requested.connect(lambda item_id, i=item: self.show_diagnosis_page(item_id))
            card.edit_requested.connect(lambda item_id, i=item: self.edit_item(item_id))
            card.delete_requested.connect(lambda item_id: self.delete_item(item_id))
            self.item_grid_layout.addWidget(card, row, col)
            
            col += 1
            if col >= 3:  # Max 3 columns
                col = 0
                row += 1

    @Slot()
    def open_add_item_dialog(self):
        dialog = AddItemDialog(self)
        dialog.item_added.connect(self.add_new_item)
        dialog.exec()

    @Slot(Item)
    def add_new_item(self, new_item: Item):
        self.items.append(new_item)
        self.save_items()
        self.refresh_dashboard()
        
    @Slot(str)
    def show_diagnosis_page(self, item_id: str):
        item_to_show = next((item for item in self.items if item.id == item_id), None)
        if item_to_show:
            self.diagnosis_page.set_item(item_to_show)
            self.stacked_widget.setCurrentWidget(self.diagnosis_page)
    
    @Slot(str, list)
    def save_diagnosis_to_item(self, item_id: str, suggested_parts: List[str]):
        """Save diagnosis entry to item's history."""
        item = next((item for item in self.items if item.id == item_id), None)
        if not item:
            return
        
        # Get the conversation from diagnosis page
        conversation = self.diagnosis_page.conversation_history
        issues_reported = conversation[1]['content'] if len(conversation) > 1 else "N/A"
        
        diagnosis_entry = DiagnosisEntry(
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            issuesReported=issues_reported,
            suggestedParts=suggested_parts,
            outcome="Diagnosis completed"
        )
        
        item.diagnosisHistory.append(diagnosis_entry)
        self.save_items()
        self.refresh_dashboard()

    @Slot()
    def show_dashboard(self):
        self.stacked_widget.setCurrentWidget(self.dashboard_widget)
        
    @Slot(str)
    def edit_item(self, item_id: str):
        """Open edit dialog for the specified item."""
        item_to_edit = next((item for item in self.items if item.id == item_id), None)
        if not item_to_edit:
            return
        
        dialog = EditItemDialog(item_to_edit, self)
        dialog.item_updated.connect(self.update_item)
        dialog.exec()
    
    @Slot(Item)
    def update_item(self, updated_item: Item):
        """Update the item in the list and refresh the display."""
        # Find and update the item in the list
        for i, item in enumerate(self.items):
            if item.id == updated_item.id:
                self.items[i] = updated_item
                break
        
        self.save_items()
        self.refresh_dashboard()
        
    @Slot(str)
    def delete_item(self, item_id: str):
        item_to_delete = next((item for item in self.items if item.id == item_id), None)
        if not item_to_delete:
            return

        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete '{item_to_delete.name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.items = [item for item in self.items if item.id != item_id]
            self.save_items()
            self.refresh_dashboard()

    def save_items(self):
        try:
            with open(DATA_FILE, 'w') as f:
                items_as_dicts = [asdict(item) for item in self.items]
                json.dump(items_as_dicts, f, indent=4)
        except IOError as e:
            QMessageBox.critical(self, "Save Error", f"Could not save data to {DATA_FILE}:\n{e}")

    def load_items(self):
        try:
            with open(DATA_FILE, 'r') as f:
                items_as_dicts = json.load(f)
                self.items = []
                for data in items_as_dicts:
                    # Convert diagnosisHistory dicts back to DiagnosisEntry objects
                    if 'diagnosisHistory' in data:
                        data['diagnosisHistory'] = [
                            DiagnosisEntry(**entry) for entry in data['diagnosisHistory']
                        ]
                    self.items.append(Item(**data))
        except FileNotFoundError:
            self.items = []  # No data file yet, this is normal
        except (json.JSONDecodeError, TypeError) as e:
            QMessageBox.warning(self, "Load Error", f"Could not load data from {DATA_FILE}. It might be corrupt.\n{e}")
            self.items = []

# ==============================================================================
# --- Application Entry Point ---
# ==============================================================================
if __name__ == "__main__":
    print("Starting Asclepius AI...")
    app = QApplication(sys.argv)
    print("QApplication created")

    # Load and apply stylesheet
    try:
        with open("style.qss", "r") as f:
            app.setStyleSheet(f.read())
        print("Stylesheet loaded")
    except FileNotFoundError:
        print("Warning: style.qss not found. Using default application style.")

    print("Creating main window...")
    window = MainWindow()
    print("Showing window...")
    window.show()
    print("Window shown, entering event loop...")
    sys.exit(app.exec())
