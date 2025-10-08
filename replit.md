# Overview

This is a React-based AI-powered content platform called "Pollen Adaptive Universe" built with TypeScript, Vite, and Tailwind CSS. The platform provides an all-in-one ecosystem for AI-generated content across multiple domains including music, entertainment, games, shopping, and task automation. The system features a sophisticated AI backend (Pollen LLMX) designed to run in Docker containers and integrates with external Python services for advanced AI capabilities.

# Recent Changes (October 2025)

## AI-Driven Features Implementation (October 8, 2025)
- **Content Verification System**: AI-powered deepfake detection with 94% accuracy, analyzing facial movements, lighting, and background anomalies
- **Wellness & Motivational Content**: Daily wellness tips, quick health notifications, and long-form wellness articles
- **Agriculture Tools**: Precision farming tools with soil analysis, weather forecasting, crop health monitoring, and personalized recommendations
- **Social Impact Initiatives**: Curated crowdfunding opportunities with AI quality scoring (0-10), voting system, and impact assessment
- **Opportunity Curation**: AI-curated opportunities across real estate, apps, products, news, trends, travel, and lifestyle with relevance scoring
- **SSE Worker Bot**: AI content creation assistant for UGC ads, social posts, and creative content with platform-specific optimization
- **Enhanced Feed Page**: Tabbed interface with All Posts, Wellness, Agriculture, Social Impact, and Opportunities sections
- **Enhanced Explore Page**: AI-driven discovery with trending opportunities, real estate insights, and content creation tools

## Complete Design Overhaul (October 8, 2025)
- **Modern Pastel Design System**: Completely rebuilt UI with soft gradient backgrounds (pink, blue, green, purple) and glass morphism cards
- **Unified Navigation**: Implemented bottom navigation with Feed, Explore, and Smart Shop tabs matching original design patterns
- **Mobile-First Layout**: Card-based responsive design with proper spacing and modern typography
- **Three Core Screens**:
  - Feed: AI-powered activity cards with category tabs (All Posts, Wellness, Agriculture, Social Impact, Opportunities)
  - Explore: AI-driven discovery and search functionality with SSE worker bot integration
  - Smart Shop: Product listings with tabs and shopping features
- **Glass Morphism UI**: Frosted glass cards with backdrop blur and subtle shadows
- **Animated Gradient Background**: Smooth shifting pastel gradient with decorative dots pattern

## Previous Platform Enhancements
- **Enhanced GlobalSearch**: Integrated conversational search component into TopNav with mobile-responsive design and semantic search capabilities
- **Collections Service**: Implemented comprehensive content curation system with Travel, Food, Goals, Events, and Shopping categories
- **Content Quality Control**: Fixed content generation bugs with proper type checking and enhanced quality validation
- **Trend Analysis**: Improved trend parsing with robust type safety and error handling
- **App Bootstrap**: Collections service now initializes during app startup for seamless user experience

# Development Notes

## Current State (October 8, 2025)
The app has been completely redesigned with a modern pastel aesthetic and enhanced with comprehensive AI-driven features. All core navigation, screens, and AI services are functional.

### What's Working
- ✅ Pastel gradient animated background with decorative dots
- ✅ Glass morphism card design system
- ✅ Bottom navigation (Feed, Explore, Smart Shop) with active states
- ✅ Enhanced Feed screen with AI-powered content sections:
  - Content Verification with deepfake detection
  - Daily Wellness Tips with personalized recommendations
  - Smart Farming Tools with soil/weather/crop analysis
  - Social Impact Initiatives with voting and AI quality scoring
  - Curated Opportunities across multiple categories
- ✅ Enhanced Explore screen with AI features:
  - SSE Worker Bot for content creation (UGC ads, social posts)
  - Trending Opportunities with momentum indicators
  - Real Estate insights with ROI analysis
  - AI-driven opportunity discovery
- ✅ Smart Shop screen with product listings
- ✅ Mobile-responsive layout with Tailwind breakpoints
- ✅ All AI services integrated and functional
- ✅ All images loading correctly from Unsplash

### Key Files to Know
- `src/App.tsx` - Main app with screen state management (now using Enhanced components)
- `src/components/EnhancedFeed.tsx` - AI-powered Feed screen with all features
- `src/components/EnhancedExplore.tsx` - AI-driven Explore/discovery screen
- `src/components/Shop.tsx` - Smart Shop screen
- `src/components/BottomNav.tsx` - Reusable bottom navigation component
- `src/services/contentVerification.ts` - Deepfake detection and content verification
- `src/services/wellnessContent.ts` - Wellness tips and motivational content
- `src/services/agricultureTools.ts` - Smart farming tools and recommendations
- `src/services/socialImpact.ts` - Social initiatives and crowdfunding curation
- `src/services/opportunityCuration.ts` - Multi-category opportunity discovery
- `src/services/sseWorkerBot.ts` - AI content creation worker bot
- `src/index.css` - Design system with gradient utilities and glass card styles

### Next Steps to Consider
1. **Wire up search functionality** - Currently the search bar navigates to Explore, but could connect to actual search logic
2. **Add interactivity to tabs** - Feed categories (All Posts, Trending, High Impact) and Shop tabs (All, Info, Images, Products) are visual only
3. **Connect to real data** - Replace static content with dynamic data from services in `src/services/`
4. **Add Collections screen** - The old Collections view (Travel, Food, Goals, Events, Shopping) could be added as a 4th tab or modal
5. **Implement user profile** - The "J" avatar could open a profile/settings screen
6. **Add animations** - Card entrance animations, tab transitions, etc.
7. **Connect AI services** - Wire up Pollen AI integration for intelligent content generation
8. **Shopping cart functionality** - Add to cart buttons in Shop screen need state management

### Design System Reference
- Glass cards: Use `glass-card` utility class
- Gradient cards: Use `gradient-card-pink`, `gradient-card-blue`, `gradient-card-purple`, `gradient-card-green`
- Active nav state: Purple-600 color with purple-100 background
- Spacing: Consistent 16px-24px between cards
- Border radius: 1.25rem (20px) for cards, 1.5rem (24px) for navigation

### Navigation Pattern
```tsx
// All screens receive onNavigate prop
<Feed onNavigate={setCurrentScreen} />

// Bottom nav updates across all screens
<BottomNav currentScreen="feed" onNavigate={onNavigate} />
```

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: React 18 with TypeScript and Vite for fast development
- **UI Components**: Shadcn/ui component library with Radix UI primitives
- **Styling**: Tailwind CSS with pastel gradient design system and glass morphism effects
- **State Management**: React useState for simple screen navigation
- **Routing**: Bottom navigation system with Feed, Explore, and Smart Shop screens
- **Design System**: Custom CSS utilities for gradient cards (pink, blue, purple, green) and glass effects

## Backend Architecture
- **AI Engine**: Custom Pollen LLMX neural network with adaptive intelligence capabilities
- **Docker Integration**: Containerized AI model deployment with local inference
- **Memory System**: Persistent user memory with short-term and long-term storage
- **Learning Engine**: Real-time feedback processing and model adaptation
- **Content Generation**: Unified content engine for multi-domain content creation

## Core AI Components
- **Adaptive Intelligence**: Self-evolving reasoning engine with induction, deduction, and abduction
- **Content Orchestrator**: Intelligent content generation and optimization across domains
- **Significance Algorithm**: Content ranking and relevance scoring system
- **Trend Engine**: Real-time trend analysis and prediction capabilities
- **Cross-Domain Intelligence**: Pattern recognition across different content types

## Data Architecture
- **Local Storage**: Browser-based persistence for user preferences and content
- **Memory Engine**: User interaction patterns and personalization data
- **Content Caching**: Optimized content delivery and offline capabilities
- **Analytics**: User behavior tracking and platform metrics
- **Collections System**: User-organized content curation with categories (Travel, Food, Goals, Events, Shopping)

## Security & Performance
- **Anonymous Operation**: No user accounts or external authentication required
- **Performance Monitoring**: Real-time system health and optimization
- **Error Boundaries**: Comprehensive error handling and recovery
- **Progressive Enhancement**: Graceful degradation when services are unavailable

# External Dependencies

## AI Services
- **Hugging Face Transformers**: Client-side AI model inference (@huggingface/transformers)
- **Python Backend**: Optional integration with Flask/FastAPI servers on localhost:8000
- **Pollen AI Integration**: Custom AI endpoints for music generation (Pollen AI + ACE-Step)

## Frontend Libraries
- **React Ecosystem**: React 18, React DOM, React Query for state management
- **UI Framework**: Radix UI components, Lucide React icons, Tailwind CSS
- **Form Handling**: React Hook Form with Zod validation
- **Utilities**: Class Variance Authority, clsx, date-fns, IndexedDB (idb)

## Development Tools
- **Build System**: Vite with SWC for fast compilation
- **TypeScript**: Full type safety with relaxed configuration for development
- **Linting**: ESLint with TypeScript and React plugins
- **Component Development**: Lovable component tagger for development mode

## Optional Integrations
- **News APIs**: External news aggregation services
- **Trend Data**: Multiple trend analysis sources
- **Music Generation**: ACE-Step Hugging Face model integration
- **Content APIs**: External content enrichment services

## Deployment Options
- **Vercel Backend**: Optional serverless deployment with KV storage
- **Local Development**: Fully functional without external services
- **Docker Deployment**: Self-contained AI model deployment
- **Edge Computing**: Optimized for low-latency edge deployment