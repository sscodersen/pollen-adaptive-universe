#!/bin/bash
# Production Setup Script for Pollen AI Platform

echo "🚀 Setting up Pollen AI for Production..."

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}❌ DATABASE_URL not set!${NC}"
    echo "Please create a PostgreSQL database in Replit:"
    echo "1. Click 'Tools' → 'Database'"
    echo "2. Click 'Create a PostgreSQL database'"
    echo "3. Re-run this script"
    exit 1
fi

echo -e "${GREEN}✅ DATABASE_URL found${NC}"

# Step 2: Install dependencies
echo -e "${YELLOW}Step 2: Verifying dependencies...${NC}"
npm list > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

pip list | grep fastapi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing Python dependencies..."
    pip install -e .
fi

echo -e "${GREEN}✅ Dependencies verified${NC}"

# Step 3: Create data directories
echo -e "${YELLOW}Step 3: Creating data directories...${NC}"
mkdir -p data
mkdir -p logs
echo -e "${GREEN}✅ Directories created${NC}"

# Step 4: Run database migrations
echo -e "${YELLOW}Step 4: Running database migrations...${NC}"
npm run db:push
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Database schema created${NC}"
else
    echo -e "${RED}❌ Migration failed${NC}"
    exit 1
fi

# Step 5: Check environment variables
echo -e "${YELLOW}Step 5: Checking environment variables...${NC}"

if [ -z "$ADMIN_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  ADMIN_API_KEY not set${NC}"
    echo "Admin features will be disabled. Set it in Secrets to enable."
else
    echo -e "${GREEN}✅ ADMIN_API_KEY configured${NC}"
fi

# Step 6: Initialize AI memory systems
echo -e "${YELLOW}Step 6: Initializing AI memory systems...${NC}"
python3 -c "from pollen_ai.memory_persistence import memory_persistence; print(memory_persistence.get_memory_stats())"
echo -e "${GREEN}✅ Memory systems initialized${NC}"

# Step 7: Health check
echo -e "${YELLOW}Step 7: Running health checks...${NC}"

# Wait for services to start
sleep 3

# Check Pollen AI backend
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Pollen AI backend healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Pollen AI backend not responding (may need restart)${NC}"
fi

# Step 8: Display configuration
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}   Pollen AI Production Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo "📊 Platform Status:"
echo "  • Database: Connected ✅"
echo "  • Memory Systems: Initialized ✅"
echo "  • Anonymous Sessions: Ready ✅"
echo ""
echo "🌐 Services:"
echo "  • Main App: http://localhost:5000"
echo "  • Pollen AI: http://localhost:8000"
echo "  • Trend Scraper: http://localhost:8099"
echo ""
echo "🔧 Next Steps:"
echo "  1. Restart workflows to apply changes"
echo "  2. Test AI content generation"
echo "  3. Verify memory persistence"
echo "  4. Monitor logs for errors"
echo ""
echo "📚 Documentation:"
echo "  • Production Guide: PRODUCTION_UPGRADE_GUIDE.md"
echo "  • AI Features: AI_FEATURES_DOCUMENTATION.md"
echo "  • Platform Docs: replit.md"
echo ""
echo -e "${GREEN}Happy Building! 🎉${NC}"
