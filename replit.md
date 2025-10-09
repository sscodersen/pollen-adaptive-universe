# Overview

This project, "Pollen Adaptive Universe," is a React-based, AI-powered content platform designed to provide an all-in-one ecosystem for AI-generated content across various domains like music, entertainment, games, shopping, and task automation. Built with TypeScript, Vite, and Tailwind CSS, it features a sophisticated AI backend (Pollen LLMX) intended for Docker deployment and integrates with external Python services for advanced AI capabilities. The platform aims to offer a modern, intuitive user experience with a focus on AI ethics, community building, and personalized content discovery.

# User Preferences

- **Communication Style**: Simple, everyday language
- **Design Preferences**: 
  - Professional black and white color scheme with subtle blue accents (no purple, pink, or excessive gold)
  - Smooth, fluid animations inspired by modern mobile apps
  - Clean, minimal UI with emphasis on functionality

# System Architecture

## Frontend Architecture
- **Framework**: React 18 with TypeScript and Vite.
- **UI/UX Design**: Professional black and white color scheme with subtle blue accents, glass morphism effects, smooth animations, and a mobile-first, card-based responsive layout. Features include dark/light mode toggle, professional button systems, optimized tab navigation, and enhanced visual hierarchy with smooth transitions.
- **Components**: Shadcn/ui, Radix UI primitives, custom utility classes for design consistency (e.g., `glass-card`, `professional-button-primary`).
- **State Management**: React `useState` for screen navigation.
- **Navigation**: Top navigation bar with tabs for Feed, Explore, Community, Smart Shop, Health Research, and Ethics Forum. Bottom navigation removed for cleaner UI.
- **Animations**: Smooth fade-in, slide-in, and scale animations with staggered effects for lists. All interactive elements have fluid hover and click feedback.

## Backend Architecture
- **AI Engine**: Custom Pollen LLMX neural network with adaptive intelligence capabilities, designed for containerized deployment (Docker).
- **AI Core Components**: Adaptive Intelligence (self-evolving reasoning), Content Orchestrator, Significance Algorithm (content ranking), Trend Engine, and Cross-Domain Intelligence.
- **Memory System**: Persistent user memory (short-term and long-term) and learning engine for model adaptation.
- **API**: Express.js endpoints for ethics reporting, transparency logs, bias statistics, community management, and moderation.

## System Design
- **Key Features**:
    - **AI-Powered Feed**: Dynamic content sections including content verification (deepfake detection), wellness tips, smart farming tools, social impact initiatives (crowdfunding with AI scoring), and curated opportunities.
    - **AI-Driven Explore**: SSE Worker Bot for content creation, trending opportunities, real estate insights, and AI-driven discovery.
    - **Community & AI Ethics**: AI transparency dashboard (decision logs, bias detection), ethical concerns reporting, AI-driven user matching, support groups, and moderation tools.
    - **Smart Shop**: Product listings and shopping features.
    - **Health & Wellness Research** (NEW - Oct 2025):
        - **Data Submission**: Multi-category health data forms (fitness, nutrition, mental health, sleep, medical) with privacy controls
        - **Research Dashboard**: AI-driven insights, trend analysis, and data correlations visualization
        - **Wellness Journey Tracker**: Goal setting, progress tracking, and milestone management for wellness journeys
        - **Research Community**: Contributors stats, top researchers, and collaborative research initiatives
    - **AI Ethics & Responsible Innovation Forum** (NEW - Oct 2025):
        - **Forum System**: Topic-based discussions with categories (AI bias, privacy, transparency, fairness, accountability, safety)
        - **Voting System**: Upvoting for topics and posts with vote count display
        - **Expert Badges**: Verification system for experts, researchers, and regular users
        - **Guidelines**: Community-driven ethical principles for AI development
        - **Transparency Dashboard**: AI decision logs and bias statistics integrated view
- **Data Architecture**: Browser-based local storage, memory engine for personalization, content caching, and analytics. Comprehensive database schema (10+ tables) for users, communities, posts, ethical reports, bias detection, and moderation. New tables for health data, wellness journeys, research findings, and forum topics.
- **Security & Performance**: Anonymous operation, real-time performance monitoring, error boundaries, and progressive enhancement.
- **User Experience Enhancements** (Oct 2025):
  - **Welcome Onboarding**: 4-step tutorial flow shown on first visit
  - **Help & Support System**: Comprehensive FAQ, video tutorials, search functionality, and contact options
  - **Real-time Weather**: Live weather data integration via Open-Meteo API
  - **Smooth Animations**: Fade-in, slide-in, and scale animations with proper timing functions
  - **Theme Toggle**: Dark/light mode with smooth transitions using next-themes

# External Dependencies

## AI Services
- **Hugging Face Transformers**: For client-side AI model inference.
- **Python Backend**: Optional integration with Flask/FastAPI servers (localhost:8000).
- **Pollen AI Integration**: Custom AI endpoints, including ACE-Step for music generation.

## Frontend Libraries
- **React Ecosystem**: React 18, React DOM, React Query.
- **UI Framework**: Radix UI components, Lucide React icons, Tailwind CSS.
- **Form Handling**: React Hook Form with Zod validation.
- **Utilities**: Class Variance Authority, clsx, date-fns, IndexedDB (idb).

## Development Tools
- **Build System**: Vite with SWC.
- **TypeScript**: Full type safety.
- **Linting**: ESLint with TypeScript and React plugins.

## Optional Integrations
- **News APIs**: External news aggregation.
- **Trend Data**: Multiple trend analysis sources.
- **Content APIs**: External content enrichment services.