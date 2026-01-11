# 🚀 Deploy Backend to Vercel - Complete UI Guide

## Step-by-Step Instructions

### 1️⃣ Push to GitHub

```bash
cd /Users/udipta18/.gemini/antigravity/scratch/voice-chatbot/backend

# Initialize git
git init
git add .
git commit -m "Initial backend commit"

# Create repo on GitHub.com (name it: voice-chatbot-backend)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/voice-chatbot-backend.git
git branch -M main
git push -u origin main
```

### 2️⃣ Deploy on Vercel

1. **Go to [vercel.com](https://vercel.com)**
   - Click "Sign Up" (use GitHub login)

2. **Import Project**
   - Click "Add New..." → "Project"
   - Find your `voice-chatbot-backend` repo
   - Click "Import"

3. **Configure Settings**
   - Framework Preset: **Other**
   - Root Directory: **.**
   - Build Command: (leave empty)
   - Output Directory: (leave empty)

4. **Add Environment Variable**
   - Click "Environment Variables"
   - Name: `OPENAI_API_KEY`
   - Value: Your actual OpenAI API key
   - Click "Add"

5. **Deploy**
   - Click "Deploy" button
   - Wait 1-2 minutes
   - Done! You'll get a URL like: `https://voice-chatbot-backend.vercel.app`

### 3️⃣ Test Your Deployment

Visit these URLs to test:
- Health check: `https://your-app.vercel.app/health`
- API docs: `https://your-app.vercel.app/docs`
- Test chat: 
  ```bash
  curl -X POST https://your-app.vercel.app/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"What time is it?","history":[]}'
  ```

### 4️⃣ Update Frontend

Update your frontend `.env.local`:
```
NEXT_PUBLIC_API_URL=https://your-app.vercel.app
```

Then restart frontend:
```bash
cd ../frontend
npm run dev
```

## 🎉 Done!

Your backend is now live on Vercel!

## 📝 Notes

- ✅ 100% FREE (100GB bandwidth/month)
- ✅ Auto-deploys on every git push
- ✅ HTTPS included
- ✅ Global CDN
- ⚠️ 10-second timeout (usually fine for APIs)

## 🔄 To Update

Just push to GitHub:
```bash
git add .
git commit -m "Update"
git push
```

Vercel will automatically redeploy!

## 🆘 Troubleshooting

### "Build failed"
- Make sure `vercel.json` exists in backend folder
- Check `requirements.txt` is correct

### "Environment variable not found"
- Go to Vercel dashboard → Settings → Environment Variables
- Make sure `OPENAI_API_KEY` is added

### "API not responding"
- Check Vercel logs in dashboard
- Make sure the URL is correct
- Try `/health` endpoint first
