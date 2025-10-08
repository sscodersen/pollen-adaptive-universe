# Overview

This project, "Pollen Adaptive Universe," is a React-based, AI-powered content platform designed to provide an all-in-one ecosystem for AI-generated content across various domains like music, entertainment, games, shopping, and task automation. Built with TypeScript, Vite, and Tailwind CSS, it features a sophisticated AI backend (Pollen LLMX) intended for Docker deployment and integrates with external Python services for advanced AI capabilities. The platform aims to offer a modern, intuitive user experience with a focus on AI ethics, community building, and personalized content discovery.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: React 18 with TypeScript and Vite.
- **UI/UX Design**: Modern pastel gradient design system with glass morphism effects, animated backgrounds, and a mobile-first, card-based responsive layout. Features include professional button systems, optimized tab navigation, and enhanced visual hierarchy.
- **Components**: Shadcn/ui, Radix UI primitives, custom utility classes for design consistency (e.g., `glass-card`, `professional-button-primary`).
- **State Management**: React `useState` for screen navigation.
- **Routing**: Bottom navigation with Feed, Explore, Community, and Smart Shop screens.

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
- **Data Architecture**: Browser-based local storage, memory engine for personalization, content caching, and analytics. Comprehensive database schema (10+ tables) for users, communities, posts, ethical reports, bias detection, and moderation.
- **Security & Performance**: Anonymous operation, real-time performance monitoring, error boundaries, and progressive enhancement.

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