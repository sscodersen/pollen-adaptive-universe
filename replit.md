# Pollen Adaptive Universe - Platform Documentation

## Overview
Pollen Universe is an AI-powered social platform featuring wellness, entertainment, news, shopping, music, and trending content. It leverages a custom AI backend for content generation and personalization, aiming to provide a dynamic and engaging user experience across various domains. The platform is built with React and TypeScript, emphasizing real-time capabilities, community interaction, and a mobile-first design.

## User Preferences
### Development Workflow
- All features use AI-generated content via **custom Pollen AI** (Absolute Zero Reasoner architecture)
- **Anonymous-first platform** - No user authentication, privacy-focused design
- **Database-backed sessions** - Persistent anonymous session tracking via PostgreSQL
- **AI Memory Persistence** - Episodic, long-term, and contextual memory systems
- Comprehensive error handling and fallback mechanisms
- Mobile-responsive design patterns throughout
- Production-ready infrastructure with job queue and monitoring

### Code Style
- TypeScript for type safety
- React hooks for state management
- Shadcn/ui for consistent components
- Tailwind CSS for styling
- Service-based architecture for business logic

## System Architecture
Pollen Universe employs a service-based architecture with distinct frontend and backend components. The frontend, built with React and TypeScript, utilizes Shadcn/ui for consistent UI components and Tailwind CSS for styling. It features dedicated pages and components for News, Entertainment, Wellness, Music, Trends, and Community, each supported by specific services for business logic (e.g., `newsDigest.ts`, `watchParty.ts`). Global state management is handled via `AppContext.tsx`.

The backend comprises a FastAPI-based Pollen AI service (`pollen_ai_optimized.py`) for all AI-powered content generation and an Express.js local backend (`local-backend.cjs`) for community, ethics, and general content APIs. A proxy server (`start.cjs`) acts as the main entry point, routing requests and managing backend services.

Key architectural decisions include:
- **AI Integration**: A custom Pollen AI backend with **Absolute Zero Reasoner** architecture (deterministic embeddings, no external models), episodic, long-term, and contextual memory systems with database persistence, and reinforcement learning. AI content generation is centralized through `pollenAIUnified.ts`.
- **Session Management**: Database-backed anonymous sessions (`server/dbSessionManager.ts`) for persistent user tracking without personal data.
- **Memory Persistence**: AI memories stored in PostgreSQL (`ai_memory_episodes`, `ai_memory_longterm`, `ai_memory_contextual`) with file-based fallback.
- **Real-time Features**: Server-Sent Events (SSE) for real-time trend scraping, WebSockets for chat and notifications, and features like Watch Parties.
- **Data Storage**: PostgreSQL database for all persistent data (content, communities, AI memory, sessions). Client-side storage using LocalStorage for preferences, IndexedDB for larger datasets, and Session storage for temporary data. Database schema includes 60+ tables for comprehensive functionality.
- **Performance Optimization**: Includes edge computing, LRU caching, lazy loading, code splitting, image optimization, request batching, and response compression.
- **UI/UX**: Emphasis on responsive, mobile-first design, consistent component library (Shadcn/ui), and AI-powered personalization for content recommendations and user feeds.
- **Gamification System**: Integrated points, badges, and leaderboards across the platform.

## Pollen AI Features (v4.0.0-AbsoluteZero)

### Core AI Capabilities
- **Text Processing**: Wellness tips, product descriptions, entertainment content, social posts, news articles
- **Audio Processing**: Music generation, speech recognition, voice assistants (models in place)
- **Image Processing**: Product images, entertainment visuals, 3D imaging (models in place)
- **Video Processing**: Entertainment videos, educational content, game footage (models in place)
- **Code Generation**: Code assistance, debugging, software development
- **Game Creation**: 3D modeling, game mechanics, full game development concepts

### Smart Home Management (NEW)
- Device control and automation across 10+ device types
- Energy usage tracking and AI-powered optimization
- Room-based device management
- Automated rule creation with AI suggestions
- Real-time device status monitoring

### Robot Management (NEW)
- Multi-robot fleet management (mobile, manipulator, drone, humanoid)
- Intelligent task planning and auto-assignment
- A* path planning with obstacle avoidance
- Battery and status monitoring
- AI-powered task optimization suggestions

### Synthetic Data Generation (NEW)
- Continuous learning through synthetic data generation
- Multi-domain training data (text, audio, image, code, game)
- Balanced batch generation for model fine-tuning
- Quality-scored samples with metadata
- Statistics tracking and monitoring

### API Endpoints
All features accessible via FastAPI backend (`pollen_ai_optimized.py`):
- `/generate` - Core AI content generation
- `/smart-home/*` - Smart home device control and automation
- `/robot/*` - Robot fleet and task management  
- `/synthetic-data/generate/*` - Training data generation
- `/reasoner/*` - Memory systems and advanced reasoning
- `/optimization/*` - Performance statistics and cache management

## External Dependencies
- **Pollen AI Backend**: Custom, in-house developed AI service (FastAPI-based).
- **Express.js**: Used for the local backend API server.
- **Shadcn/ui**: UI component library.
- **Lucide React**: Icon library.
- **React 18 + Vite**: Frontend framework and build tool.
- **Tailwind CSS**: Utility-first CSS framework.
- **IndexedDB**: Client-side database for larger datasets.
- **Exploding Topics**: Source for real-time trend data via SSE.