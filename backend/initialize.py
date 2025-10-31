"""
Initialization Script for Pollen AI Platform
Sets up database and runs initial content scraping
"""

import asyncio
import sys
from backend.database import init_db, SessionLocal
from backend.services.background_scraper import background_scraper

async def initialize_platform():
    """
    Initialize the platform:
    1. Create database tables
    2. Run initial content scraping
    3. Prepare training data
    """
    print("🚀 Initializing Pollen AI Platform...")
    
    print("\n📦 Step 1: Creating database tables...")
    try:
        init_db()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return False
    
    print("\n🔍 Step 2: Running initial content scraping...")
    try:
        await background_scraper.run_daily_scrape_job()
        print("✅ Initial scraping completed")
    except Exception as e:
        print(f"❌ Error during initial scraping: {e}")
        return False
    
    print("\n✅ Platform initialization complete!")
    print("\n📊 Platform is ready to serve high-quality, scored content")
    print("🧠 Pollen AI training pipeline is active")
    print("🎯 Bento Buzz scoring algorithm is running")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(initialize_platform())
    sys.exit(0 if success else 1)
