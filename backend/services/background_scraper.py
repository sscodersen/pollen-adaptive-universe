"""
Background Scraper Job Service
Handles automated daily content scraping, scoring, and storage
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
import logging

from backend.database import SessionLocal
from backend.services.enhanced_scraper import EnhancedScraperService
from backend.services.content_storage import content_storage
from backend.services.pollen_ai_trainer import pollen_ai_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundScraperService:
    def __init__(self):
        self.scraper = EnhancedScraperService()
        self.is_running = False
        self.last_run = None
    
    async def run_daily_scrape_job(self):
        """
        Main daily scraping job - runs all scraping tasks
        """
        logger.info("🚀 Starting daily content scraping job...")
        
        db = SessionLocal()
        try:
            job_id = content_storage.create_scraper_job(db, "daily_scrape")
            
            total_scraped = 0
            total_scored = 0
            total_passed = 0
            
            news_results = await self._scrape_and_store_news(db)
            total_scraped += news_results["scraped"]
            total_passed += news_results["passed"]
            
            trends_results = await self._scrape_and_store_trends(db)
            total_scraped += trends_results["scraped"]
            total_passed += trends_results["passed"]
            
            events_results = await self._scrape_and_store_events(db)
            total_scraped += events_results["scraped"]
            total_passed += events_results["passed"]
            
            products_results = await self._scrape_and_store_products(db)
            total_scraped += products_results["scraped"]
            total_passed += products_results["passed"]
            
            total_scored = total_scraped
            
            content_storage.update_scraper_job(
                db,
                job_id,
                status="completed",
                items_scraped=total_scraped,
                items_scored=total_scored,
                items_passed=total_passed
            )
            
            logger.info(f"✅ Daily scrape complete: {total_scraped} items scraped, {total_passed} passed quality threshold")
            
            if total_passed >= 10:
                logger.info("🧠 Starting Pollen AI training with high-quality content...")
                await self._train_pollen_ai(db)
            
            self.last_run = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Daily scrape job failed: {e}")
            if 'job_id' in locals():
                content_storage.update_scraper_job(
                    db,
                    job_id,
                    status="failed",
                    error_message=str(e)
                )
        finally:
            db.close()
    
    async def _scrape_and_store_news(self, db) -> Dict[str, int]:
        """Scrape and store news articles"""
        logger.info("📰 Scraping news articles...")
        try:
            articles = await self.scraper._scrape_hacker_news()
            rss_articles = await self.scraper._scrape_rss_feeds()
            all_articles = articles + rss_articles
            
            passed = 0
            for article in all_articles:
                result = content_storage.store_content_with_score(db, article, "news")
                if result:
                    passed += 1
            
            logger.info(f"📰 News: {len(all_articles)} scraped, {passed} passed threshold")
            return {"scraped": len(all_articles), "passed": passed}
        except Exception as e:
            logger.error(f"Error scraping news: {e}")
            return {"scraped": 0, "passed": 0}
    
    async def _scrape_and_store_trends(self, db) -> Dict[str, int]:
        """Scrape and store trending topics"""
        logger.info("📈 Scraping trending topics...")
        try:
            trends = await self.scraper.scrape_exploding_topics(max_results=30)
            
            passed = 0
            for trend in trends:
                result = content_storage.store_content_with_score(db, trend, "trend")
                if result:
                    passed += 1
            
            logger.info(f"📈 Trends: {len(trends)} scraped, {passed} passed threshold")
            return {"scraped": len(trends), "passed": passed}
        except Exception as e:
            logger.error(f"Error scraping trends: {e}")
            return {"scraped": 0, "passed": 0}
    
    async def _scrape_and_store_events(self, db) -> Dict[str, int]:
        """Scrape and store events"""
        logger.info("📅 Scraping events...")
        try:
            events = await self.scraper._scrape_events_from_web()
            
            passed = 0
            for event in events:
                result = content_storage.store_content_with_score(db, event, "event")
                if result:
                    passed += 1
            
            logger.info(f"📅 Events: {len(events)} scraped, {passed} passed threshold")
            return {"scraped": len(events), "passed": passed}
        except Exception as e:
            logger.error(f"Error scraping events: {e}")
            return {"scraped": 0, "passed": 0}
    
    async def _scrape_and_store_products(self, db) -> Dict[str, int]:
        """Scrape and store product information"""
        logger.info("📦 Scraping products...")
        try:
            products = await self.scraper._scrape_products_from_web()
            
            passed = 0
            for product in products:
                result = content_storage.store_content_with_score(db, product, "product")
                if result:
                    passed += 1
            
            logger.info(f"📦 Products: {len(products)} scraped, {passed} passed threshold")
            return {"scraped": len(products), "passed": passed}
        except Exception as e:
            logger.error(f"Error scraping products: {e}")
            return {"scraped": 0, "passed": 0}
    
    async def _train_pollen_ai(self, db):
        """Train Pollen AI with high-quality content"""
        try:
            training_dataset = content_storage.get_training_dataset(
                db,
                min_quality=80.0,
                max_items=500,
                unused_only=True
            )
            
            if not training_dataset:
                logger.info("No new training data available")
                return
            
            logger.info(f"Training Pollen AI with {len(training_dataset)} high-quality examples")
            
            training_batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            training_data = {
                "examples": [
                    {
                        "text": f"{item['title']}. {item['description']}",
                        "metadata": {
                            "category": item["category"],
                            "score": item["quality_score"],
                            "source": item["source"],
                            "keywords": item["title"].lower().split()
                        }
                    }
                    for item in training_dataset
                ],
                "quality_threshold": 80.0,
                "batch_id": training_batch_id
            }
            
            content_ids = [item["content_id"] for item in training_dataset]
            content_storage.mark_training_data_used(db, content_ids, training_batch_id)
            
            logger.info(f"✅ Training data prepared: {len(training_dataset)} examples")
            
        except Exception as e:
            logger.error(f"Error training Pollen AI: {e}")
    
    async def start_periodic_scraping(self, interval_hours: int = 24):
        """
        Start periodic scraping at specified interval
        """
        self.is_running = True
        logger.info(f"🔄 Starting periodic scraping every {interval_hours} hours")
        
        while self.is_running:
            await self.run_daily_scrape_job()
            await asyncio.sleep(interval_hours * 3600)
    
    def stop(self):
        """Stop periodic scraping"""
        self.is_running = False
        logger.info("⏸️  Stopped periodic scraping")


background_scraper = BackgroundScraperService()
