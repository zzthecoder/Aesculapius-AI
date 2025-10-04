"""
Parts Lookup Module for FixMate
Uses web scraping and search APIs to find replacement parts
"""

import requests
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import json

# ==============================================================================
# --- Parts Search Functions ---
# ==============================================================================

def search_parts_duckduckgo(item_name: str, part_description: str) -> List[Dict[str, str]]:
    """
    Search for parts using DuckDuckGo's instant answer API (no API key needed).
    Returns a list of search results with titles and URLs.
    """
    query = f"{item_name} {part_description} replacement part buy"
    encoded_query = quote_plus(query)
    
    try:
        # DuckDuckGo instant answer API
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Extract related topics
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics'][:5]:  # Limit to 5 results
                if isinstance(topic, dict) and 'Text' in topic and 'FirstURL' in topic:
                    results.append({
                        'title': topic['Text'],
                        'url': topic['FirstURL'],
                        'source': 'DuckDuckGo'
                    })
        
        return results
    except Exception as e:
        print(f"Error searching DuckDuckGo: {e}")
        return []

def search_ifixit_guides(item_name: str) -> List[Dict[str, str]]:
    """
    Search iFixit for repair guides (using their public API).
    No API key required for basic searches.
    """
    try:
        # iFixit search endpoint
        encoded_query = quote_plus(item_name)
        url = f"https://www.ifixit.com/api/2.0/search/{encoded_query}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if 'results' in data:
            for guide in data['results'][:5]:  # Limit to 5 results
                if guide.get('type') == 'guide':
                    results.append({
                        'title': guide.get('title', 'Unknown Guide'),
                        'url': guide.get('url', ''),
                        'difficulty': guide.get('difficulty', 'Unknown'),
                        'source': 'iFixit'
                    })
        
        return results
    except Exception as e:
        print(f"Error searching iFixit: {e}")
        return []

def generate_shopping_links(item_name: str, part_description: str) -> List[Dict[str, str]]:
    """
    Generate direct shopping links for major retailers.
    """
    query = f"{item_name} {part_description}"
    encoded_query = quote_plus(query)
    
    shopping_sites = [
        {
            'name': 'Amazon',
            'url': f"https://www.amazon.com/s?k={encoded_query}",
            'icon': '🛒'
        },
        {
            'name': 'eBay',
            'url': f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}",
            'icon': '🏪'
        },
        {
            'name': 'AliExpress',
            'url': f"https://www.aliexpress.com/wholesale?SearchText={encoded_query}",
            'icon': '🌐'
        },
        {
            'name': 'Google Shopping',
            'url': f"https://www.google.com/search?tbm=shop&q={encoded_query}",
            'icon': '🔍'
        }
    ]
    
    return shopping_sites

def search_parts_comprehensive(item_name: str, part_description: str) -> Dict[str, any]:
    """
    Comprehensive parts search combining multiple sources.
    Returns a dictionary with guides, search results, and shopping links.
    """
    return {
        'guides': search_ifixit_guides(item_name),
        'search_results': search_parts_duckduckgo(item_name, part_description),
        'shopping_links': generate_shopping_links(item_name, part_description)
    }

# ==============================================================================
# --- Optional: Scrapy-based Advanced Scraping ---
# ==============================================================================

def get_scrapy_spider_template():
    """
    Returns a template for a Scrapy spider for advanced scraping.
    This can be used if the user wants to implement custom scrapers.
    """
    template = '''
import scrapy

class PartsSpider(scrapy.Spider):
    name = 'parts_spider'
    
    def __init__(self, item_name='', part_description='', *args, **kwargs):
        super(PartsSpider, self).__init__(*args, **kwargs)
        self.item_name = item_name
        self.part_description = part_description
        self.start_urls = [
            f'https://www.example-parts-site.com/search?q={item_name}+{part_description}'
        ]
    
    def parse(self, response):
        # Extract part information
        for part in response.css('.part-item'):
            yield {
                'name': part.css('.part-name::text').get(),
                'price': part.css('.part-price::text').get(),
                'url': part.css('a::attr(href)').get(),
            }
'''
    return template

# ==============================================================================
# --- Test Function ---
# ==============================================================================

if __name__ == "__main__":
    # Test the parts lookup
    print("Testing Parts Lookup System...")
    print("\n" + "="*60)
    
    test_item = "Samsung Washing Machine"
    test_part = "door latch"
    
    print(f"Searching for: {test_item} - {test_part}")
    print("="*60 + "\n")
    
    results = search_parts_comprehensive(test_item, test_part)
    
    print("📚 REPAIR GUIDES (iFixit):")
    for guide in results['guides']:
        print(f"  • {guide['title']}")
        print(f"    Difficulty: {guide['difficulty']}")
        print(f"    URL: {guide['url']}\n")
    
    print("\n🔍 SEARCH RESULTS:")
    for result in results['search_results']:
        print(f"  • {result['title']}")
        print(f"    URL: {result['url']}\n")
    
    print("\n🛒 SHOPPING LINKS:")
    for link in results['shopping_links']:
        print(f"  {link['icon']} {link['name']}: {link['url']}")
