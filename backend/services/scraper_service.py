import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, AsyncGenerator
import json
import re

class ScraperService:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    async def search_and_scrape(self, query: str, max_results: int = 5) -> AsyncGenerator[str, None]:
        """
        Search the web and scrape results, streaming them back
        """
        try:
            search_results = await self._search_web(query)
            
            for i, result in enumerate(search_results[:max_results]):
                yield json.dumps({
                    "type": "search_result",
                    "index": i + 1,
                    "total": min(len(search_results), max_results),
                    "data": result
                }) + "\n"
                
                content = await self._scrape_url(result.get("url", ""))
                
                if content:
                    processed = self._process_content(content)
                    yield json.dumps({
                        "type": "scraped_content",
                        "index": i + 1,
                        "url": result.get("url", ""),
                        "data": processed
                    }) + "\n"
                    
        except Exception as e:
            yield json.dumps({"error": f"Scraper error: {str(e)}"}) + "\n"
    
    async def _search_web(self, query: str) -> List[Dict]:
        """
        Perform web search (placeholder - would integrate with search API)
        """
        results = [
            {
                "title": f"Result for {query} - Source 1",
                "url": "https://example.com/1",
                "snippet": f"This is a relevant result for {query}..."
            },
            {
                "title": f"Result for {query} - Source 2",
                "url": "https://example.com/2",
                "snippet": f"Another relevant result for {query}..."
            }
        ]
        return results
    
    async def _scrape_url(self, url: str) -> str:
        """
        Scrape content from a URL
        """
        try:
            async with httpx.AsyncClient(timeout=30.0, headers=self.headers) as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]
        except Exception:
            return ""
    
    def _process_content(self, content: str) -> Dict:
        """
        Process and clean scraped content
        """
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return {
            "summary": ' '.join(sentences[:3]) if sentences else content[:200],
            "full_text": content,
            "word_count": len(content.split()),
            "sentence_count": len(sentences)
        }

scraper_service = ScraperService()
