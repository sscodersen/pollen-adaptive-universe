# Pollen AI Vercel Backend

## Quick Deploy Instructions

### 1. Setup
1. Create a new GitHub repository
2. Push this `backend/` folder contents to the repository
3. Go to [vercel.com](https://vercel.com) and import the repository

### 2. Environment Variables
In Vercel Dashboard → Settings → Environment Variables, add:

```
POLLEN_AI_ENDPOINT=http://localhost:8000
POLLEN_AI_API_KEY=your-pollen-ai-key
```

### 3. Create KV Database
1. In Vercel project → Storage tab
2. Create KV Database
3. Link to your project (auto-adds KV_REST_API_URL and KV_REST_API_TOKEN)

### 4. Update Frontend
In `src/services/pollenAI.ts`, update the production URL:
```javascript
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://YOUR-VERCEL-URL.vercel.app/api'  // Replace with your actual URL
  : 'http://localhost:3000/api';
```

### 5. Deploy
Your backend will automatically deploy on each git push to main branch.

## API Endpoints
- `POST /api/ai/generate` - Generate AI content
- `GET /api/content/feed` - Fetch generated content

## Local Development
```bash
cd backend
npm install
npx vercel dev
```