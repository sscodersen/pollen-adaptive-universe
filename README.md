# 🚀 New Frontier AI Platform - Powered by Pollen AI

An innovative, privacy-first AI assistant platform that leverages Pollen AI to provide personalized, adaptive experiences across multiple domains including shopping, travel, content creation, smart home management, health, and education.

## ✨ Features

### Core Capabilities
- **🛍️ Shopping Assistant**: AI-powered product recommendations with real-time SSE streaming
- **✈️ Travel Planner**: Personalized trip planning and itinerary generation
- **📰 News Aggregator**: Unbiased news from diverse sources
- **✍️ Content Generator**: Create articles, blog posts, and marketing copy
- **🏠 Smart Home Management**: Control and automate IoT devices
- **💪 Health & Wellness**: Personalized health advice and wellness plans
- **📚 Education Assistant**: Adaptive learning paths and tutoring

### Technical Features
- **Real-time SSE Streaming**: All AI responses stream in real-time for immediate feedback
- **Floating Search Bar**: Global search accessible via Cmd/Ctrl+K
- **Privacy-First**: No authentication required, completely anonymous
- **Responsive Design**: Mobile-optimized UI with beautiful gradients
- **Error Handling**: Production-ready error boundaries and loading states
- **Web Scraper Fallback**: Automatic web scraping when AI lacks information

## 🏗️ Architecture

### Frontend
- **Framework**: React 18 + Vite
- **UI Library**: Chakra UI with custom theming
- **Routing**: React Router v7
- **State Management**: React Hooks
- **Icons**: Lucide React
- **Styling**: Emotion + Framer Motion

### Backend
- **Framework**: FastAPI
- **Streaming**: Server-Sent Events (SSE)
- **HTTP Client**: HTTPX (async)
- **Database**: PostgreSQL with SQLAlchemy (optional)
- **Web Scraping**: BeautifulSoup4
- **Environment**: Python 3.11+

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm 9+
- Python 3.11+
- Your Pollen AI model URL (or OpenAI API key as fallback)

### 1. Clone and Install

```bash
# Install frontend dependencies
npm install

# Install backend dependencies (if not using uv)
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```bash
# Copy the example
cp .env.example .env
```

**Edit with your Pollen AI configuration:**
```env
# Your Pollen AI Model URL (REQUIRED)
AI_MODEL_URL=https://your-pollen-ai-model.com/api/stream

# Optional: OpenAI fallback
OPENAI_API_KEY=your-openai-key-here

# Backend
PORT=8000
HOST=0.0.0.0

# Frontend
VITE_API_URL=http://localhost:8000
```

### 3. Run Development Servers

Both workflows should start automatically in Replit. If running locally:

**Terminal 1 - Backend:**
```bash
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

Visit the webview to see your platform!

## 📦 Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for comprehensive deployment instructions.

### Quick Deploy Summary

1. **Frontend (Vercel)**: 
   - Push to GitHub
   - Import to Vercel
   - Set `VITE_API_URL` environment variable
   - Deploy!

2. **Backend (Render/Railway)**:
   - Connect GitHub repo
   - Set environment variables
   - Deploy!

## 🎯 Pollen AI Integration

Your Pollen AI model should implement this API contract:

### Request Format
```json
POST /api/stream
{
  "prompt": "User's question or request",
  "context": {},
  "stream": true
}
```

### Response Format (SSE)
```
Content-Type: text/event-stream

data: {"text": "First chunk"}
data: {"text": " of the response"}
data: {"text": " streaming live"}
```

## 🛠️ Project Structure

```
pollen-ai-platform/
├── src/
│   ├── components/
│   │   ├── common/          # Reusable components
│   │   │   ├── ErrorBoundary.jsx
│   │   │   ├── FloatingSearchBar.jsx
│   │   │   ├── LoadingState.jsx
│   │   │   └── ErrorState.jsx
│   │   └── layout/          # Layout components
│   ├── features/            # Feature modules
│   │   ├── dashboard/       # Enhanced personalized dashboard
│   │   ├── shopping/
│   │   ├── travel/
│   │   ├── news/
│   │   ├── content/
│   │   ├── smarthome/
│   │   ├── health/
│   │   └── education/
│   ├── hooks/              # Custom React hooks (SSE streaming)
│   ├── services/           # API services
│   └── utils/              # Utilities
├── backend/
│   ├── routers/            # API endpoints (all use SSE)
│   ├── services/           # Business logic
│   │   ├── ai_service.py   # AI integration
│   │   └── scraper_service.py
│   ├── database.py         # Database config
│   └── main.py            # FastAPI app
├── public/                # Static assets
├── .env.example           # Environment template
├── vercel.json           # Vercel deployment config
├── requirements.txt      # Python dependencies
├── DEPLOYMENT.md         # Detailed deployment guide
└── package.json          # Node dependencies
```

## 🌐 API Endpoints

All endpoints support SSE streaming for real-time responses.

### Shopping
- `GET /api/shopping/search` - Product recommendations
- `GET /api/shopping/categories` - Available categories

### Travel
- `GET /api/travel/plan` - Trip planning
- `GET /api/travel/destinations` - Popular destinations

### News
- `GET /api/news/fetch` - Fetch and summarize news
- `GET /api/news/categories` - News categories

### Content
- `GET /api/content/generate` - Generate content
- `GET /api/content/types` - Content types

### Smart Home
- `GET /api/smarthome/control` - Device control
- `GET /api/smarthome/devices` - Device types

### Health
- `GET /api/health/advice` - Health advice
- `GET /api/health/categories` - Health categories

### Education
- `GET /api/education/learn` - Learning assistance
- `GET /api/education/subjects` - Available subjects

### AI & Scraper
- `POST /api/ai/chat` - General AI chat
- `POST /api/scraper/search` - Web scraping

## ✨ Key Features Implemented

### Floating Global Search
Press `Cmd/Ctrl + K` anywhere to open the intelligent search that:
- Auto-detects your intent
- Routes to the appropriate feature
- Provides suggested searches
- Beautiful animations and UX

### Enhanced Dashboard
- Personalized greeting based on time of day
- Recent activity tracking
- Trending topics
- Beautiful gradient design

### Production-Ready
- Error boundaries for graceful error handling
- Loading states for better UX
- Optimized build with code splitting
- Environment-based configuration
- Comprehensive deployment documentation

## 🔒 Security & Privacy

- **No Authentication**: Completely anonymous, no sign-up required
- **Privacy-First**: No user tracking or data collection by default
- **CORS Enabled**: Configurable for production domains
- **Environment Variables**: Sensitive data stored securely

## 📊 Performance Optimizations

- **Code Splitting**: Automatic chunk splitting for vendors (React, Chakra UI, Icons)
- **Lazy Loading**: Routes and components loaded on demand
- **Tree Shaking**: Unused code eliminated in production builds
- **Minification**: Production builds optimized with Terser
- **Caching**: Static assets cached with proper headers
- **Console Removal**: Console logs removed in production

## 🐛 Troubleshooting

### Frontend not connecting to backend
- Verify `VITE_API_URL` in `.env`
- Check backend is running on port 8000
- Ensure CORS is properly configured

### SSE streaming not working
- Verify your Pollen AI endpoint returns proper SSE format
- Check browser console for connection errors
- Ensure EventSource is supported in your browser

### Build errors
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf .vite`
- Check Node.js version: `node --version` (requires 18+)

## 🌟 Vision Alignment

This platform implements the "New Frontier of the Internet" vision:

✅ **Adaptive User Interface** - Floating search bar, personalized dashboard
✅ **SSE Streaming** - Real-time AI responses across all features  
✅ **Content Generation & Curation** - AI-powered content creation
✅ **Smart Home Integration** - IoT device management
✅ **Educational Features** - Adaptive learning paths
✅ **Health & Wellness** - Personalized health guidance
✅ **Privacy-First** - No authentication, anonymous usage
✅ **Production-Ready** - Vercel deployment configured
✅ **Extensible** - Easy to add new features

## 📝 Next Steps

1. Add your Pollen AI model URL to `.env`
2. Test all features with your AI model
3. Customize branding and colors in `src/theme/index.js`
4. Deploy to production (see DEPLOYMENT.md)
5. Add custom domain (optional)

---

**Built with ❤️ using Pollen AI - The New Frontier of the Internet**