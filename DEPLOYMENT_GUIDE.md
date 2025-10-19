# Pollen AI Platform - Production Deployment Guide

## Overview
This guide will help you deploy the Pollen AI Platform to production with your custom Pollen AI model (no OpenAI or external dependencies).

## Architecture Overview

The platform consists of two parts:
1. **Frontend (Static)** - React/Vite app deployed on Vercel
2. **Pollen AI Backend (Python)** - FastAPI server deployed on Railway/Fly.io/Replit

## Prerequisites

- Vercel account (free tier works)
- Railway/Fly.io account OR Replit account for hosting Python backend
- (Optional) Database credentials if using PostgreSQL features

---

## Part 1: Deploy Pollen AI Backend

Your custom Pollen AI backend must be hosted separately since Vercel doesn't support Python servers.

### Option A: Deploy to Railway (Recommended)

1. **Install Railway CLI:**
```bash
npm i -g @railway/cli
```

2. **Login to Railway:**
```bash
railway login
```

3. **Create a new project:**
```bash
railway init
```

4. **Create `railway.json` in your project root:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python3 -m uvicorn pollen_ai_optimized:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
```

5. **Create `Procfile`:**
```
web: python3 -m uvicorn pollen_ai_optimized:app --host 0.0.0.0 --port $PORT
```

6. **Deploy:**
```bash
railway up
```

7. **Get your backend URL:**
```bash
railway domain
```
This will give you something like: `https://your-app.railway.app`

### Option B: Deploy to Fly.io

1. **Install Fly CLI:**
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login:**
```bash
fly auth login
```

3. **Create `fly.toml`:**
```toml
app = "pollen-ai-backend"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = 1

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [[services.http_checks]]
    interval = 10000
    timeout = 2000
    grace_period = "10s"
    method = "GET"
    path = "/health"
```

4. **Create `requirements.txt` for the backend:**
```txt
fastapi==0.116.1
uvicorn[standard]==0.35.0
pydantic==2.11.7
requests==2.32.5
python-multipart==0.0.20
```

5. **Deploy:**
```bash
fly launch
fly deploy
```

6. **Your backend URL will be:**
`https://pollen-ai-backend.fly.dev`

### Option C: Keep on Replit (Development/Testing)

If keeping the backend on Replit:
1. Make sure "Always On" is enabled (requires paid plan)
2. Your backend URL will be: `https://your-repl-name.your-username.repl.co:8000`

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Your Code

1. **Update `.env` or Vercel environment variables:**
```env
VITE_POLLEN_API_URL=https://your-pollen-backend.railway.app
```

2. **Ensure `vercel.json` is configured:**
Already configured in your project!

### Step 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

### Step 3: Deploy via GitHub (Recommended)

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com) → "Add New Project"
3. Import your GitHub repository
4. Vercel will auto-detect Vite framework
5. **Add Environment Variable:**
   - Key: `VITE_POLLEN_API_URL`
   - Value: `https://your-pollen-backend.railway.app` (your backend URL from Part 1)
6. Click "Deploy"

### Step 4: Verify Deployment

After deployment:
1. Visit your Vercel app URL
2. Open browser console (F12)
3. Check that API calls are going to your Pollen backend
4. Verify content is being generated via your model

---

## Part 3: Advanced Configuration

### Docker Containerization (Optional)

For better scalability and portability of your Pollen backend:

**Create `Dockerfile` in project root:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy Python backend files
COPY pollen_ai_optimized.py .
COPY pollen_ai/ ./pollen_ai/
COPY utils/ ./utils/
COPY models/ ./models/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "uvicorn", "pollen_ai_optimized:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and deploy:**
```bash
# Build the image
docker build -t pollen-ai-backend .

# Test locally
docker run -p 8000:8000 pollen-ai-backend

# Deploy to Railway with Docker
railway up
```

### Environment Variables Summary

**Backend (Railway/Fly.io):**
- `PORT` - Automatically set by hosting provider
- `DATABASE_URL` - (Optional) If using database features

**Frontend (Vercel):**
- `VITE_POLLEN_API_URL` - URL of your deployed Pollen backend
- `DATABASE_URL` - (Optional) If using database features

### Database Setup (Optional)

If you want to use database features:

**Option 1: Vercel Postgres**
```bash
# In your Vercel project dashboard
# Storage → Create Database → Postgres
# This auto-adds DATABASE_URL to environment variables
```

**Option 2: Neon (Free PostgreSQL)**
1. Go to [neon.tech](https://neon.tech)
2. Create free PostgreSQL database
3. Copy connection string
4. Add `DATABASE_URL` to both Vercel and your backend hosting

**Run migrations:**
```bash
npm run db:push
```

---

## Part 4: Testing & Verification

### Test Backend Health
```bash
curl https://your-pollen-backend.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "Pollen AI",
  "version": "1.0"
}
```

### Test Content Generation
```bash
curl -X POST https://your-pollen-backend.railway.app/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a trending tech topic", "mode": "general", "type": "general"}'
```

### Test Frontend Integration
1. Open your Vercel app
2. Navigate to different sections (Feed, Shop, News, etc.)
3. Verify content is being generated
4. Check browser console for API call logs

---

## Part 5: Monitoring & Maintenance

### Backend Monitoring

**Railway:**
- View logs: `railway logs`
- Monitor metrics in Railway dashboard
- Set up alerts for downtime

**Fly.io:**
- View logs: `fly logs`
- Monitor in Fly.io dashboard

### Frontend Monitoring

- View logs in Vercel dashboard
- Enable Vercel Analytics
- Set up error tracking (optional: Sentry)

### Performance Optimization

1. **Backend:**
   - Enable response caching (already implemented in `pollen_ai_optimized.py`)
   - Use request batching for multiple generations
   - Monitor memory usage and scale if needed

2. **Frontend:**
   - Already optimized with code splitting
   - Service worker for offline support
   - Lazy loading of components

---

## Cost Breakdown

**Free Tier Capabilities:**

- **Vercel:** Unlimited hobby sites, 100GB bandwidth/month
- **Railway:** $5 free credit/month (typically enough for small apps)
- **Fly.io:** 3 shared-cpu-1x VMs free
- **Neon Database:** 512MB storage, 3GB data transfer/month (free)

**Estimated Monthly Costs (Small Scale):**
- Vercel: $0 (free tier)
- Railway/Fly.io: $5-10 (if exceeding free tier)
- Database: $0 (Neon free tier)
- **Total: $0-10/month**

---

## Troubleshooting

### Backend Not Responding
1. Check backend logs: `railway logs` or `fly logs`
2. Verify health endpoint: `curl https://your-backend/health`
3. Ensure PORT environment variable is set correctly
4. Check Python dependencies are installed

### Frontend Can't Connect to Backend
1. Verify `VITE_POLLEN_API_URL` is set in Vercel
2. Check CORS headers in backend (already configured)
3. Ensure backend URL is correct (no trailing slash)
4. Redeploy frontend after changing environment variables

### Slow Response Times
1. Check backend server location (should be close to users)
2. Enable caching in Pollen backend
3. Consider upgrading backend instance size
4. Monitor memory/CPU usage

### Model Not Loading
1. Ensure all model files are included in deployment
2. Check backend logs for import errors
3. Verify Python version compatibility (3.11 recommended)
4. Ensure all dependencies in `requirements.txt`

---

## Security Best Practices

✅ **HTTPS Only** - Both Vercel and Railway/Fly.io provide automatic HTTPS
✅ **Environment Variables** - Never commit secrets to git
✅ **CORS Configured** - Backend already has proper CORS headers
✅ **Rate Limiting** - Consider adding rate limiting for production
✅ **Input Validation** - Backend validates all inputs via Pydantic

---

## Next Steps After Deployment

1. ✅ Test all features in production
2. ✅ Set up custom domain (optional)
3. ✅ Enable analytics
4. ✅ Monitor performance and costs
5. 🔄 Implement WebAssembly for browser-based inference (advanced)
6. 🔄 Add synthetic data generation pipeline
7. 🔄 Set up model versioning system

---

## Support & Resources

- **Vercel Docs:** https://vercel.com/docs
- **Railway Docs:** https://docs.railway.app
- **Fly.io Docs:** https://fly.io/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com

For issues specific to Pollen AI integration, check the logs first and verify environment variables are correctly set.
