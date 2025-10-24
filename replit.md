# New Frontier of the Internet - AI Platform

## Overview
A comprehensive AI-powered platform with mobile-inspired UI, featuring SSE (Server-Sent Events) streaming for real-time AI responses across multiple services including shopping, travel, news, content generation, and more. Built with React + Vite frontend and FastAPI backend.

## Project Setup (Updated - October 24, 2025)

### Frontend
- **Framework**: React 18 with Vite 4
- **UI Library**: Chakra UI with custom theme
- **Routing**: React Router DOM
- **Icons**: Lucide React
- **Port**: 5000 (configured for Replit environment)
- **Host**: 0.0.0.0 with HMR configuration and proxy

### Backend
- **Framework**: FastAPI with Python 3.11
- **SSE**: sse-starlette for Server-Sent Events
- **AI Integration**: Configurable AI model URL (Pollen AI) with OpenAI fallback
- **Scraper**: BeautifulSoup for web scraping
- **Port**: 8000 (localhost only, proxied through Vite)

## Architecture

### Frontend Structure
```
src/
├── components/
│   ├── layout/          # MainLayout, Header, BottomNavigation
│   └── common/          # FeatureCard, SearchBar
├── features/            # Feature modules (dashboard, shopping, travel, etc.)
├── hooks/               # useSSEStream for SSE handling
├── services/            # API configuration
├── theme/               # Chakra UI custom theme
└── utils/               # Constants and utilities
```

### Backend Structure
```
backend/
├── main.py             # FastAPI app with CORS
├── routers/            # API endpoints (shopping, travel, news, content, scraper, ai)
├── services/           # AI service, scraper service
├── models/             # Pydantic schemas
└── utils/              # Utilities
```

## Key Features

### Implemented
1. **Mobile-Inspired UI**
   - Gradient backgrounds with glassmorphism effects
   - Personalized header ("Hey Jane, Welcome back!")
   - Bottom navigation with icons
   - Feature cards for each service

2. **SSE Streaming**
   - EventSource API for real-time streaming
   - useSSEStream React hook for easy integration
   - Mock data fallback when AI not configured

3. **API Proxy**
   - Vite proxy routes `/api` to backend on port 8000
   - Solves Replit port accessibility issues
   - Seamless frontend-backend communication

4. **Features (in development)**
   - Shopping Assistant (Sister) - Product recommendations
   - Travel Planner - Trip planning and itineraries
   - News Aggregator - Unbiased news from multiple sources
   - Content Generator - AI-powered content creation
   - Smart Home - IoT device management
   - Health & Wellness - Personalized wellness plans
   - Education - Adaptive learning paths

### Pending
- PostgreSQL database setup
- User authentication and personalization
- Complete implementation of all feature modules
- Deployment configuration

## Configuration

### Vite (vite.config.js)
- Port 5000 for frontend
- Path aliases for clean imports
- Proxy `/api` requests to `http://127.0.0.1:8000`
- HMR configuration for Replit

### FastAPI (backend/main.py)
- CORS middleware for development
- Modular routers for each feature
- SSE streaming endpoints (GET requests for EventSource compatibility)

## Environment Variables
- `AI_MODEL_URL`: Custom AI model endpoint (optional)
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `DATABASE_URL`: PostgreSQL connection string (pending setup)

## Recent Changes
- **October 24, 2025**: Platform transformation
  - Complete UI redesign to mobile-inspired interface
  - Implemented React Router with multiple routes
  - Created custom Chakra UI theme with gradients and glassmorphism
  - Built FastAPI backend with SSE streaming
  - Fixed SSE implementation to use EventSource API
  - Added Vite proxy for API requests
  - Created reusable components and hooks
  - Implemented Shopping Assistant with live SSE streaming

## Development Workflow
1. **Frontend**: `npm run dev` (configured as Frontend workflow)
2. **Backend**: `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000` (configured as Backend workflow)

## User Preferences
- Modern, mobile-inspired design
- SSE streaming for real-time responses
- Support for custom AI models (Pollen AI)
- Fallback to scraper bot when AI lacks information
- Comprehensive feature set across multiple domains
