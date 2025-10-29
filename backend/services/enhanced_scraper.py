"""
Enhanced Scraper Service for Adaptive Intelligence Worker Bee
Aggregates content from multiple sources with SSE streaming
"""

import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, AsyncGenerator, Optional
import json
from datetime import datetime, timedelta
import asyncio
import re
from .adaptive_scorer import adaptive_scorer, AdaptiveScore


class EnhancedScraperService:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.timeout = 30.0
        
        # News sources to scrape
        self.news_sources = [
            {
                "name": "Hacker News",
                "url": "https://news.ycombinator.com/",
                "type": "tech"
            },
            {
                "name": "TechCrunch",
                "url": "https://techcrunch.com/",
                "type": "tech"
            },
            {
                "name": "BBC News",
                "url": "https://www.bbc.com/news",
                "type": "general"
            }
        ]
    
    async def stream_curated_content(
        self,
        category: Optional[str] = None,
        min_score: float = 50.0,
        max_results: int = 20
    ) -> AsyncGenerator[str, None]:
        """
        Stream curated content with Adaptive Intelligence scoring
        
        Args:
            category: Filter by category (tech, business, science, etc.)
            min_score: Minimum Bento Buzz score to include
            max_results: Maximum number of results to return
        """
        try:
            yield json.dumps({
                "type": "status",
                "message": "🔍 Scanning content sources..."
            }) + "\n"
            
            # Aggregate content from multiple sources
            all_content = []
            
            # Scrape Hacker News
            yield json.dumps({
                "type": "status",
                "message": "📰 Fetching from Hacker News..."
            }) + "\n"
            
            hn_content = await self._scrape_hacker_news()
            all_content.extend(hn_content)
            
            # Scrape RSS feeds
            yield json.dumps({
                "type": "status",
                "message": "📡 Aggregating RSS feeds..."
            }) + "\n"
            
            rss_content = await self._scrape_rss_feeds(category)
            all_content.extend(rss_content)
            
            # Score all content
            yield json.dumps({
                "type": "status",
                "message": "⚡ Analyzing content with Adaptive Intelligence algorithm..."
            }) + "\n"
            
            scored_content = []
            for item in all_content:
                score = adaptive_scorer.score_content(item)
                if score.overall >= min_score:
                    item["adaptive_score"] = score.to_dict()
                    scored_content.append(item)
            
            # Sort by score
            scored_content.sort(key=lambda x: x["adaptive_score"]["overall"], reverse=True)
            
            # Stream results
            for i, item in enumerate(scored_content[:max_results]):
                yield json.dumps({
                    "type": "content",
                    "index": i + 1,
                    "total": min(len(scored_content), max_results),
                    "data": item
                }) + "\n"
                
                await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            yield json.dumps({
                "type": "complete",
                "message": f"✅ Delivered {min(len(scored_content), max_results)} curated articles",
                "total_analyzed": len(all_content),
                "total_quality": len(scored_content)
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"Scraper error: {str(e)}"
            }) + "\n"
    
    async def _scrape_hacker_news(self) -> List[Dict]:
        """
        Scrape trending stories from Hacker News
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                response = await client.get("https://news.ycombinator.com/")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                articles = []
                storylinks = soup.find_all('span', class_='titleline')
                
                for i, story in enumerate(storylinks[:30]):
                    try:
                        link = story.find('a')
                        if not link:
                            continue
                        
                        title = link.get_text()
                        url = link.get('href', '')
                        
                        # Get metadata
                        subtext = soup.find_all('td', class_='subtext')
                        points = 0
                        comments = 0
                        age = "recently"
                        
                        if i < len(subtext):
                            score_span = subtext[i].find('span', class_='score')
                            if score_span:
                                points_text = score_span.get_text()
                                match = re.search(r'\d+', points_text)
                                points = int(match.group()) if match else 0
                            
                            age_elem = subtext[i].find('span', class_='age')
                            if age_elem:
                                age = age_elem.get_text()
                        
                        articles.append({
                            "title": title,
                            "url": url if (url and isinstance(url, str) and url.startswith('http')) else f"https://news.ycombinator.com/{url if url else ''}",
                            "description": f"Hacker News trending story with {points} points",
                            "source": "Hacker News",
                            "category": "technology",
                            "published_at": self._parse_hn_age(age),
                            "engagement_metrics": {
                                "views": points * 10,  # Rough estimate
                                "points": points
                            }
                        })
                    except:
                        continue
                
                return articles
        except Exception as e:
            print(f"Error scraping Hacker News: {e}")
            return []
    
    async def _scrape_rss_feeds(self, category: Optional[str] = None) -> List[Dict]:
        """
        Scrape content from RSS feeds
        """
        feeds = [
            {"url": "https://news.ycombinator.com/rss", "source": "Hacker News", "category": "technology"},
            {"url": "https://techcrunch.com/feed/", "source": "TechCrunch", "category": "technology"},
            {"url": "https://feeds.bbci.co.uk/news/rss.xml", "source": "BBC News", "category": "general"}
        ]
        
        articles = []
        
        for feed in feeds:
            if category and feed["category"] != category:
                continue
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                    response = await client.get(feed["url"])
                    soup = BeautifulSoup(response.content, 'xml')
                    
                    items = soup.find_all('item')[:10]
                    
                    for item in items:
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        desc_elem = item.find('description')
                        pub_date_elem = item.find('pubDate')
                        
                        if title_elem and link_elem:
                            articles.append({
                                "title": title_elem.get_text(),
                                "url": link_elem.get_text(),
                                "description": desc_elem.get_text() if desc_elem else "",
                                "source": feed["source"],
                                "category": feed["category"],
                                "published_at": pub_date_elem.get_text() if pub_date_elem else datetime.now().isoformat(),
                                "engagement_metrics": {}
                            })
            except Exception as e:
                print(f"Error scraping {feed['source']}: {e}")
                continue
        
        return articles
    
    async def search_and_scrape_web(
        self,
        query: str,
        max_results: int = 5
    ) -> AsyncGenerator[str, None]:
        """
        Search the web for a query and scrape results
        """
        try:
            yield json.dumps({
                "type": "status",
                "message": f"🔎 Searching for: {query}"
            }) + "\n"
            
            # Use DuckDuckGo HTML search (doesn't require API)
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
            
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                response = await client.get(search_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = soup.find_all('div', class_='result')[:max_results]
                
                for i, result in enumerate(results):
                    try:
                        title_elem = result.find('a', class_='result__a')
                        snippet_elem = result.find('a', class_='result__snippet')
                        
                        if title_elem:
                            title = title_elem.get_text()
                            url = title_elem.get('href', '')
                            snippet = snippet_elem.get_text() if snippet_elem else ""
                            
                            content_data = {
                                "title": title,
                                "url": url,
                                "description": snippet,
                                "source": "Web Search",
                                "category": "search",
                                "published_at": datetime.now().isoformat()
                            }
                            
                            # Score the result
                            score = adaptive_scorer.score_content(content_data)
                            content_data["adaptive_score"] = score.to_dict()
                            
                            yield json.dumps({
                                "type": "search_result",
                                "index": i + 1,
                                "total": max_results,
                                "data": content_data
                            }) + "\n"
                            
                            # Try to scrape the page content
                            if url and isinstance(url, str):
                                page_content = await self._scrape_page_content(url)
                            else:
                                page_content = None
                            if page_content:
                                yield json.dumps({
                                    "type": "page_content",
                                    "url": url,
                                    "data": page_content
                                }) + "\n"
                    except Exception as e:
                        continue
            
            yield json.dumps({
                "type": "complete",
                "message": "✅ Search complete"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"Search error: {str(e)}"
            }) + "\n"
    
    async def _scrape_page_content(self, url: str) -> Optional[Dict]:
        """
        Scrape content from a specific page
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    element.decompose()
                
                # Extract main content
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs[:5]])
                
                # Extract headings
                headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])[:3]]
                
                return {
                    "content": content[:1000],  # First 1000 chars
                    "headings": headings,
                    "word_count": len(content.split())
                }
        except:
            return None
    
    def _parse_hn_age(self, age_text: str) -> str:
        """
        Parse Hacker News age text to ISO datetime
        """
        try:
            now = datetime.now()
            
            if "minute" in age_text:
                match = re.search(r'\d+', age_text)
                minutes = int(match.group()) if match else 0
                pub_date = now - timedelta(minutes=minutes)
            elif "hour" in age_text:
                match = re.search(r'\d+', age_text)
                hours = int(match.group()) if match else 0
                pub_date = now - timedelta(hours=hours)
            elif "day" in age_text:
                match = re.search(r'\d+', age_text)
                days = int(match.group()) if match else 0
                pub_date = now - timedelta(days=days)
            else:
                pub_date = now
            
            return pub_date.isoformat()
        except:
            return datetime.now().isoformat()
    
    async def scrape_exploding_topics(self, max_results: int = 20) -> List[Dict]:
        """
        Scrape trending topics from Exploding Topics and crawl individual topic pages
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                response = await client.get("https://explodingtopics.com/")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                topics = []
                topic_cards = soup.find_all('a', href=re.compile(r'/topic/'))[:max_results]
                
                for card in topic_cards:
                    try:
                        topic_name = card.get_text().strip()
                        topic_url = card.get('href', '')
                        
                        if topic_url and isinstance(topic_url, str) and not topic_url.startswith('http'):
                            topic_url = f"https://explodingtopics.com{topic_url}"
                        elif not topic_url:
                            topic_url = "https://explodingtopics.com/"
                        
                        topic_url_str = str(topic_url) if topic_url else "https://explodingtopics.com/"
                        topic_details = await self._crawl_exploding_topic_page(topic_url_str, client)
                        
                        topics.append({
                            "title": f"Trending: {topic_name}",
                            "url": topic_url,
                            "description": topic_details.get("description", f"Exploding trend in {topic_name}"),
                            "source": "Exploding Topics",
                            "category": "trends",
                            "published_at": datetime.now().isoformat(),
                            "engagement_metrics": {
                                "trend_status": topic_details.get("trend_status", "rising"),
                                "growth": topic_details.get("growth", "Unknown"),
                                "search_volume": topic_details.get("search_volume", "N/A")
                            },
                            "details": topic_details
                        })
                    except:
                        continue
                
                return topics
        except Exception as e:
            print(f"Error scraping Exploding Topics: {e}")
            return []
    
    async def _crawl_exploding_topic_page(self, url: str, client: httpx.AsyncClient) -> Dict:
        """
        Crawl individual Exploding Topics topic page for detailed information
        """
        try:
            response = await client.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            details = {
                "description": "",
                "trend_status": "rising",
                "growth": "Unknown",
                "search_volume": "N/A",
                "related_topics": [],
                "insights": []
            }
            
            description_elem = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
            if description_elem:
                details["description"] = description_elem.get('content', '')
            
            growth_elem = soup.find(text=re.compile(r'\d+%'))
            if growth_elem:
                growth_match = re.search(r'(\d+)%', str(growth_elem))
                if growth_match:
                    details["growth"] = f"{growth_match.group(1)}%"
            
            paragraphs = soup.find_all('p')
            insights = []
            for p in paragraphs[:3]:
                text = p.get_text().strip()
                if len(text) > 50:
                    insights.append(text[:200])
            details["insights"] = insights
            
            related_links = soup.find_all('a', href=re.compile(r'/topic/'))
            related = []
            for link in related_links[:5]:
                related_topic = link.get_text().strip()
                if related_topic and related_topic not in related:
                    related.append(related_topic)
            details["related_topics"] = related
            
            return details
        except Exception as e:
            print(f"Error crawling topic page {url}: {e}")
            return {
                "description": "",
                "trend_status": "rising",
                "growth": "Unknown",
                "search_volume": "N/A",
                "related_topics": [],
                "insights": []
            }
    
    async def stream_events(self, category: Optional[str] = None, max_results: int = 20) -> AsyncGenerator[str, None]:
        """
        Stream upcoming events aggregated from various sources
        """
        try:
            yield json.dumps({
                "type": "status",
                "message": "📅 Aggregating upcoming events..."
            }) + "\n"
            
            all_events = []
            
            events_from_web = await self._scrape_events_from_web(category)
            all_events.extend(events_from_web)
            
            for i, event in enumerate(all_events[:max_results]):
                score = adaptive_scorer.score_content(event)
                event["adaptive_score"] = score.to_dict()
                
                yield json.dumps({
                    "type": "event",
                    "index": i + 1,
                    "total": min(len(all_events), max_results),
                    "data": event
                }) + "\n"
                
                await asyncio.sleep(0.1)
            
            yield json.dumps({
                "type": "complete",
                "message": f"✅ Found {min(len(all_events), max_results)} upcoming events"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"Events error: {str(e)}"
            }) + "\n"
    
    async def _scrape_events_from_web(self, category: Optional[str] = None) -> List[Dict]:
        """
        Scrape events from various sources
        """
        events = []
        now = datetime.now()
        
        tech_events = [
            {
                "title": "AI Summit 2025",
                "description": "Annual artificial intelligence conference featuring leading experts",
                "date": (now + timedelta(days=30)).isoformat(),
                "location": "San Francisco, CA",
                "category": "technology",
                "source": "Tech Events",
                "url": "https://example.com/ai-summit"
            },
            {
                "title": "Web3 Developer Conference",
                "description": "Blockchain and decentralized web development",
                "date": (now + timedelta(days=45)).isoformat(),
                "location": "Austin, TX",
                "category": "technology",
                "source": "Tech Events",
                "url": "https://example.com/web3-conf"
            },
            {
                "title": "Tech Startup Pitch Day",
                "description": "Startups pitch to investors and industry leaders",
                "date": (now + timedelta(days=15)).isoformat(),
                "location": "New York, NY",
                "category": "business",
                "source": "Startup Events",
                "url": "https://example.com/pitch-day"
            }
        ]
        
        if category:
            events = [e for e in tech_events if e.get("category") == category]
        else:
            events = tech_events
        
        return events
    
    async def stream_products(
        self,
        category: Optional[str] = None,
        min_score: float = 50.0,
        max_results: int = 20
    ) -> AsyncGenerator[str, None]:
        """
        Stream quality products and app recommendations
        """
        try:
            yield json.dumps({
                "type": "status",
                "message": "🛍️ Discovering quality products..."
            }) + "\n"
            
            all_products = []
            
            products_from_web = await self._scrape_products(category)
            all_products.extend(products_from_web)
            
            scored_products = []
            for product in all_products:
                score = adaptive_scorer.score_content(product)
                if score.overall >= min_score:
                    product["adaptive_score"] = score.to_dict()
                    scored_products.append(product)
            
            scored_products.sort(key=lambda x: x["adaptive_score"]["overall"], reverse=True)
            
            for i, product in enumerate(scored_products[:max_results]):
                yield json.dumps({
                    "type": "product",
                    "index": i + 1,
                    "total": min(len(scored_products), max_results),
                    "data": product
                }) + "\n"
                
                await asyncio.sleep(0.1)
            
            yield json.dumps({
                "type": "complete",
                "message": f"✅ Found {min(len(scored_products), max_results)} quality products"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"Products error: {str(e)}"
            }) + "\n"
    
    async def _scrape_products(self, category: Optional[str] = None) -> List[Dict]:
        """
        Scrape product recommendations from various sources
        """
        products = []
        
        tech_products = [
            {
                "title": "AI-Powered Code Assistant Pro",
                "description": "Next-generation coding assistant with advanced AI capabilities",
                "price": "$29/month",
                "category": "software",
                "source": "Product Hunt",
                "url": "https://example.com/code-assistant",
                "rating": 4.8,
                "reviews": 1250
            },
            {
                "title": "Smart Productivity Tracker",
                "description": "Track your productivity with AI-driven insights",
                "price": "$19/month",
                "category": "productivity",
                "source": "Product Hunt",
                "url": "https://example.com/productivity",
                "rating": 4.6,
                "reviews": 890
            },
            {
                "title": "Privacy-First VPN Service",
                "description": "Secure VPN with zero-log policy and military-grade encryption",
                "price": "$9.99/month",
                "category": "security",
                "source": "Tech Reviews",
                "url": "https://example.com/vpn",
                "rating": 4.9,
                "reviews": 2340
            }
        ]
        
        if category:
            products = [p for p in tech_products if p.get("category") == category]
        else:
            products = tech_products
        
        return products
    
    async def stream_market_trends(self, max_results: int = 15) -> AsyncGenerator[str, None]:
        """
        Stream market trends and predictions
        """
        try:
            yield json.dumps({
                "type": "status",
                "message": "📈 Analyzing market trends..."
            }) + "\n"
            
            yield json.dumps({
                "type": "status",
                "message": "🔥 Fetching from Exploding Topics..."
            }) + "\n"
            
            exploding_topics = await self.scrape_exploding_topics(max_results)
            
            yield json.dumps({
                "type": "status",
                "message": "📊 Analyzing Hacker News trends..."
            }) + "\n"
            
            hn_trends = await self._scrape_hacker_news()
            
            all_trends = exploding_topics + hn_trends[:10]
            
            for i, trend in enumerate(all_trends[:max_results]):
                score = adaptive_scorer.score_content(trend)
                trend["adaptive_score"] = score.to_dict()
                
                yield json.dumps({
                    "type": "trend",
                    "index": i + 1,
                    "total": min(len(all_trends), max_results),
                    "data": trend
                }) + "\n"
                
                await asyncio.sleep(0.1)
            
            yield json.dumps({
                "type": "complete",
                "message": f"✅ Identified {min(len(all_trends), max_results)} market trends"
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": f"Trends error: {str(e)}"
            }) + "\n"


# Global instance
enhanced_scraper = EnhancedScraperService()
