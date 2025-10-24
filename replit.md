# Nectar.js - AI Website Template

## Overview
This is a React + Vite frontend application that demonstrates Server-Sent Events (SSE) streaming with OpenAI's API. It's a template for building AI-powered websites with real-time streaming responses.

## Project Setup (Completed - October 24, 2025)
- **Framework**: React 18 with Vite 4
- **UI Library**: Chakra UI
- **Key Feature**: SSE (Server-Sent Events) for streaming OpenAI completions
- **Port**: 5000 (configured for Replit environment)
- **Host**: 0.0.0.0 with proper HMR configuration for Replit proxy

## Architecture
- **Frontend Only**: This is a client-side only application
- **API Integration**: Direct OpenAI API calls from browser using SSE
- **Styling**: Chakra UI with Emotion for styling

## Configuration
- Vite configured to:
  - Serve on 0.0.0.0:5000
  - Support Replit's proxy via HMR settings
  - Use WebSocket Secure (wss) for hot reload

## Dependencies
- React & React DOM
- Vite with SWC plugin for fast refresh
- Chakra UI for components
- SSE.js library for Server-Sent Events
- Framer Motion for animations

## Environment Variables
- `VITE_OPENAI_API_KEY`: OpenAI API key (optional - user can add if they want to use the app)

## Recent Changes
- **October 24, 2025**: Initial Replit setup
  - Updated vite.config.js for Replit environment (port 5000, 0.0.0.0 host)
  - Updated .gitignore for Node.js project
  - Installed all npm dependencies
  - Configured frontend workflow

## Notes
- The app uses the deprecated `text-davinci-003` model - may need updating to newer models
- API key is loaded from environment variables for security
