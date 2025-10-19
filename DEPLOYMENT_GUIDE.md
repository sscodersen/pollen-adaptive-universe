# Pollen AI Platform - Vercel Deployment Guide

## Overview
This guide will help you deploy the Pollen AI Platform to Vercel with full production functionality.

## Prerequisites
- Vercel account (free tier works)
- OpenAI API key for AI content generation
- (Optional) Database credentials if using PostgreSQL

## Deployment Steps

### 1. Environment Variables Setup
Add these environment variables in your Vercel project settings:

#### Required:
- `OPENAI_API_KEY` - Your OpenAI API key (get from platform.openai.com)

#### Optional (if using database features):
- `DATABASE_URL` - PostgreSQL connection string (e.g., from Vercel Postgres or Neon)

### 2. Deploy to Vercel

#### Option A: Deploy via Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy (follow prompts)
vercel

# Deploy to production
vercel --prod
```

#### Option B: Deploy via GitHub Integration
1. Push your code to a GitHub repository
2. Go to vercel.com and click "Add New Project"
3. Import your GitHub repository
4. Vercel will auto-detect the framework (Vite)
5. Add environment variables in the project settings
6. Click "Deploy"

### 3. Post-Deployment Configuration

After deployment:
1. Visit your Vercel dashboard
2. Go to Settings → Environment Variables
3. Add `OPENAI_API_KEY` with your API key
4. Redeploy the project to apply changes

### 4. Database Setup (Optional)

If you want to use database features:

#### Using Vercel Postgres:
```bash
# In your Vercel project dashboard
# Go to Storage → Create Database → Postgres
# This will automatically add DATABASE_URL to your environment variables
```

#### Using External Database (Neon, etc.):
1. Get your PostgreSQL connection string
2. Add `DATABASE_URL` environment variable in Vercel
3. Run migrations: `npm run db:push`

## Build Configuration

The project is configured with:
- **Framework**: Vite
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Node Version**: 20.x (recommended)

## Features in Production

✅ **No Demo Data** - All content is generated via OpenAI
✅ **Real-time AI Generation** - Content generated on-demand
✅ **Optimized Performance** - Production build with code splitting
✅ **Secure** - Environment variables properly configured
✅ **Scalable** - Static hosting with serverless capabilities

## Troubleshooting

### Build Fails
- Ensure all dependencies are in `package.json`
- Check that `OPENAI_API_KEY` is set
- Verify Node version is 18.x or higher

### AI Generation Not Working
- Verify `OPENAI_API_KEY` is set in Vercel environment variables
- Check OpenAI API quota and billing
- Redeploy after adding environment variables

### Database Connection Issues
- Ensure `DATABASE_URL` format is correct
- For Vercel Postgres, use the connection string from dashboard
- Check that database is accessible from Vercel's network

## Performance Optimization

The platform includes:
- Code splitting for faster initial load
- Service worker for edge caching
- Lazy loading of routes and components
- Optimized asset delivery via Vercel CDN

## Cost Considerations

**Vercel Hosting**: Free tier supports hobby projects
**OpenAI API**: Pay-per-use (estimate $0.002-0.03 per request depending on model)
**Database**: Vercel Postgres free tier: 256MB storage, or use free Neon tier

## Support

For issues or questions:
1. Check Vercel deployment logs
2. Review browser console for errors
3. Verify environment variables are set correctly

## Next Steps

After deployment:
1. Test all features in production
2. Monitor OpenAI API usage
3. Set up custom domain (optional)
4. Enable analytics in Vercel dashboard
