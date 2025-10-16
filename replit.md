# Pollen Adaptive Universe - Platform Documentation

## Overview
Pollen Universe is an AI-powered social platform featuring wellness, entertainment, news, shopping, music, and trending content. It leverages a custom AI backend for content generation and personalization, aiming to provide a dynamic and engaging user experience across various domains. The platform is built with React and TypeScript, emphasizing real-time capabilities, community interaction, and a mobile-first design.

## User Preferences
### Development Workflow
- All features use AI-generated content via Pollen AI
- No external API dependencies (OpenAI, Anthropic, etc.)
- Comprehensive error handling and fallback mechanisms
- Mobile-responsive design patterns throughout

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
- **AI Integration**: A custom Pollen AI backend with a "Zero-Start Learning" model, episodic, long-term, and contextual memory systems, and reinforcement learning. AI content generation is centralized through `pollenAIUnified.ts`.
- **Real-time Features**: Server-Sent Events (SSE) for real-time trend scraping, WebSockets for chat and notifications, and features like Watch Parties.
- **Data Storage**: Client-side storage using LocalStorage for preferences, IndexedDB for larger datasets, and Session storage for temporary data. The backend is designed to be stateless regarding user data.
- **Performance Optimization**: Includes edge computing, LRU caching, lazy loading, code splitting, image optimization, request batching, and response compression.
- **UI/UX**: Emphasis on responsive, mobile-first design, consistent component library (Shadcn/ui), and AI-powered personalization for content recommendations and user feeds.
- **Gamification System**: Integrated points, badges, and leaderboards across the platform.

## External Dependencies
- **Pollen AI Backend**: Custom, in-house developed AI service (FastAPI-based).
- **Express.js**: Used for the local backend API server.
- **Shadcn/ui**: UI component library.
- **Lucide React**: Icon library.
- **React 18 + Vite**: Frontend framework and build tool.
- **Tailwind CSS**: Utility-first CSS framework.
- **IndexedDB**: Client-side database for larger datasets.
- **Exploding Topics**: Source for real-time trend data via SSE.