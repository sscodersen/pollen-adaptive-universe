# Deployment Guide - New Frontier AI Platform (Pollen AI)

This guide will help you deploy your Pollen AI platform to production.

## Architecture Overview

- **Frontend**: React + Vite → Deploy to **Vercel**
- **Backend**: FastAPI → Deploy to **Render**, **Railway**, or **Fly.io**
- **Database**: PostgreSQL (optional) → Can use Vercel Postgres or external provider

---

## Prerequisites

1. Your Pollen AI model URL (or OpenAI API key as fallback)
2. GitHub account (for easy deployment)
3. Vercel account (free tier available)
4. Render/Railway account for backend (free tier available)

---

## Part 1: Deploy Backend (FastAPI)

### Option A: Deploy to Render (Recommended)

1. **Push your code to GitHub**
2. **Go to [Render.com](https://render.com)** and sign in
3. **Create New Web Service**
   - Connect your GitHub repository
   - Name: `pollen-ai-backend`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt` (we'll create this)
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   
4. **Add Environment Variables** in Render dashboard:
   ```
   AI_MODEL_URL=https://your-pollen-ai-model.com/api/stream
   OPENAI_API_KEY=your-openai-key-here (optional)
   PORT=8000
   ```

5. **Deploy** and copy the backend URL (e.g., `https://pollen-ai-backend.onrender.com`)

### Option B: Deploy to Railway

1. Go to [Railway.app](https://railway.app)
2. Create new project from GitHub repo
3. Select the repository
4. Add environment variables (same as above)
5. Railway will auto-detect Python and deploy
6. Copy the backend URL

### Option C: Deploy to Fly.io

1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Follow prompts and add environment variables
4. Deploy with: `fly deploy`

---

## Part 2: Deploy Frontend (React + Vite) to Vercel

### Step 1: Update Environment Variables

Create a `.env.production` file:

```bash
# Backend API URL (from Part 1)
VITE_API_URL=https://your-backend-url.onrender.com

# Environment
NODE_ENV=production
```

### Step 2: Deploy to Vercel

#### Option A: Via Vercel Dashboard (Easiest)

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Add New Project"
3. Import your GitHub repository
4. **Framework Preset**: Vite
5. **Root Directory**: `./` (leave as is)
6. **Build Command**: `npm run build`
7. **Output Directory**: `dist`
8. **Install Command**: `npm install`

9. **Add Environment Variables**:
   - Key: `VITE_API_URL`
   - Value: `https://your-backend-url.onrender.com` (from Part 1)

10. Click "Deploy"

#### Option B: Via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel --prod

# Follow prompts and add environment variables when asked
```

---

## Part 3: Connect Your Pollen AI Model

### Update Backend Environment

In your backend deployment (Render/Railway/Fly.io), set the environment variable:

```bash
AI_MODEL_URL=https://your-actual-pollen-ai-url.com/api/stream
```

### Expected API Format

Your Pollen AI model should accept POST requests:

**Request:**
```json
{
  "prompt": "User's question or request",
  "context": {},
  "stream": true
}
```

**Response:** Server-Sent Events (SSE) stream
```
data: {"text": "First chunk of response"}
data: {"text": " second chunk"}
data: {"text": " third chunk"}
```

---

## Part 4: Optional - Set Up Database

If you want to store user preferences and data:

### Vercel Postgres (Recommended for integration)

1. In Vercel dashboard, go to your project
2. Go to "Storage" tab
3. Create "Postgres" database
4. Copy the `DATABASE_URL` connection string
5. Add it to your backend environment variables on Render/Railway

### Alternative: External PostgreSQL

- **Supabase** (Free tier available)
- **Neon** (Serverless Postgres)
- **ElephantSQL** (Managed PostgreSQL)

---

## Part 5: Verify Deployment

### Test Your Platform

1. **Visit your Vercel URL**: `https://your-app.vercel.app`
2. **Test a feature** (e.g., Shopping Assistant, Travel Planner)
3. **Verify SSE streaming** is working
4. **Check backend logs** on Render/Railway for any errors

### Common Issues

**Frontend can't reach backend:**
- Verify `VITE_API_URL` is set correctly in Vercel
- Check backend CORS settings (should allow all origins or your Vercel domain)

**AI responses not working:**
- Verify `AI_MODEL_URL` is set correctly in backend environment
- Test the Pollen AI endpoint directly
- Check backend logs for errors

**SSE streaming issues:**
- Ensure your Pollen AI model returns proper SSE format
- Check browser console for connection errors
- Verify the backend is streaming correctly

---

## Part 6: Continuous Deployment

Both Vercel and Render/Railway support automatic deployments:

- **Push to GitHub** → Automatically deploys to production
- **Environment variables** are preserved across deployments
- **Rollback** available if needed

---

## Cost Estimate

- **Vercel** (Frontend): FREE for hobby projects
- **Render** (Backend): FREE tier available (spins down after inactivity)
- **Railway** (Backend): $5/month after free credits
- **Database**: FREE tier available on most providers

---

## Environment Variables Summary

### Frontend (.env.production)
```
VITE_API_URL=https://your-backend-url.com
NODE_ENV=production
```

### Backend (Render/Railway/Fly.io)
```
AI_MODEL_URL=https://your-pollen-ai-model.com/api/stream
OPENAI_API_KEY=sk-xxx (optional fallback)
PORT=8000
DATABASE_URL=postgresql://... (optional)
```

---

## Support

If you encounter issues:
1. Check backend logs in Render/Railway dashboard
2. Check browser console for frontend errors
3. Verify all environment variables are set correctly
4. Test your Pollen AI endpoint independently

---

## Next Steps

1. ✅ Deploy backend to Render/Railway
2. ✅ Deploy frontend to Vercel
3. ✅ Connect your Pollen AI model URL
4. ✅ Test all features
5. ✅ Set up custom domain (optional)
6. ✅ Enable analytics and monitoring

---

**Your platform is now live and ready to serve users! 🚀**