# Backend Deployment Guide

The Pollen AI Platform backend uses FastAPI with Server-Sent Events (SSE) for real-time AI streaming. Since SSE requires persistent connections, we recommend deploying the backend to platforms that support long-running processes.

## Recommended Backend Platforms

### Option 1: Railway (Recommended)
Railway supports long-running processes and is perfect for FastAPI with SSE.

**Steps:**
1. Create a Railway account at https://railway.app
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize project: `railway init`
5. Add environment variables:
   ```bash
   railway variables set AI_MODEL_URL="your-ai-model-url"
   railway variables set OPENAI_API_KEY="your-openai-key"
   railway variables set PORT=8000
   ```
6. Deploy: `railway up`
7. Get your backend URL from Railway dashboard
8. Update frontend `.env`:
   ```
   VITE_API_URL=https://your-railway-app.railway.app
   ```

### Option 2: Render
Render offers free tier with persistent connections.

**Steps:**
1. Create account at https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - Name: `pollen-ai-backend`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard:
   - `AI_MODEL_URL`
   - `OPENAI_API_KEY`
6. Deploy and get your URL
7. Update frontend `.env` with the Render URL

### Option 3: Fly.io
Fly.io provides excellent support for long-running connections.

**Steps:**
1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
2. Login: `flyctl auth login`
3. Create `fly.toml` in root:
   ```toml
   app = "pollen-ai-backend"
   
   [build]
   builder = "paketobuildpacks/builder:base"
   
   [[services]]
   internal_port = 8000
   protocol = "tcp"
   
   [[services.ports]]
   handlers = ["http"]
   port = 80
   
   [[services.ports]]
   handlers = ["tls", "http"]
   port = 443
   
   [env]
   PORT = "8000"
   ```
4. Deploy: `flyctl deploy`
5. Set secrets:
   ```bash
   flyctl secrets set AI_MODEL_URL="your-url"
   flyctl secrets set OPENAI_API_KEY="your-key"
   ```

## Frontend Deployment on Vercel

1. Push your code to GitHub
2. Go to https://vercel.com
3. Import your repository
4. Configure build settings:
   - Framework Preset: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`
5. Add environment variable:
   - `VITE_API_URL`: Your backend URL (from Railway/Render/Fly.io)
6. Deploy!

## Environment Variables

### Backend (.env)
```bash
AI_MODEL_URL=https://your-pollen-ai-model.com/api/stream
OPENAI_API_KEY=sk-your-openai-key-here
PORT=8000
HOST=0.0.0.0
```

### Frontend (.env)
```bash
VITE_API_URL=https://your-backend-url.com
```

## Testing Deployment

After deployment, test SSE streaming:

1. Open your frontend URL
2. Try any feature (Shopping, Travel, etc.)
3. Verify that AI responses stream in real-time
4. Check browser console for any errors

## Troubleshooting

### SSE Connection Fails
- Verify CORS is configured in backend (`backend/main.py`)
- Check `VITE_API_URL` is correct in frontend
- Ensure backend is running and accessible

### Timeout Errors
- Make sure backend platform supports long-running connections
- Avoid deploying backend to Vercel (10-60s timeout)
- Use Railway/Render/Fly.io instead

### CORS Errors
- Backend should have:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

## Cost Breakdown

- **Railway**: $5/month for hobby plan, free $5 credit
- **Render**: Free tier available (limited hours), $7/month for always-on
- **Fly.io**: Free tier with 3 shared VMs
- **Vercel (Frontend)**: Free tier sufficient for most use cases

## Performance Optimization

1. **Enable Caching**: Use Redis for caching AI responses
2. **CDN**: Vercel automatically provides CDN for frontend
3. **Database**: Add PostgreSQL for user preferences (optional)
4. **Monitoring**: Use Sentry for error tracking

## Security Best Practices

1. Never commit `.env` files
2. Use environment variables for all secrets
3. Enable HTTPS (automatic on all platforms)
4. Rotate API keys regularly
5. Implement rate limiting in production

## Next Steps

1. Deploy backend to chosen platform
2. Deploy frontend to Vercel
3. Configure custom domain (optional)
4. Set up monitoring and analytics
5. Add database for persistent storage (optional)

For support, check platform-specific documentation:
- Railway: https://docs.railway.app
- Render: https://render.com/docs
- Fly.io: https://fly.io/docs
- Vercel: https://vercel.com/docs
