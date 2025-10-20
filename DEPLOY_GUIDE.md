# 🚀 DEPLOY TO RENDER - SIMPLE GUIDE

## What You Need:
1. GitHub account (free)
2. Render account (free, NO credit card needed)

---

## 📦 STEP 1: Push to GitHub (3 minutes)

```bash
# Navigate to your project
cd "c:\Users\ROG\Desktop\thesis\THESIS 2 (REVISED)\revision(after system defense)\final\clustering_pipeline"

# Initialize git
git init
git add .
git commit -m "Ready for deployment"

# Create repo on GitHub:
# 1. Go to https://github.com/new
# 2. Name it: patient-clustering-dashboard
# 3. Click "Create repository"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/patient-clustering-dashboard.git
git branch -M main
git push -u origin main
```

---

## 🌐 STEP 2: Deploy on Render (2 minutes)

1. Go to **https://render.com**
2. Click **"Get Started"** (sign up with GitHub - FREE)
3. Click **"New +"** → **"Web Service"**
4. Find and select your repo: `patient-clustering-dashboard`
5. Render auto-fills everything:
   - **Name**: patient-clustering-dashboard
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn api:app`
6. Scroll down to **Environment Variables**:
   - Click **"Add Environment Variable"**
   - **Key**: `GEMINI_API_KEY`
   - **Value**: Your Gemini API key
7. Click **"Create Web Service"**
8. Wait 5-10 minutes ☕

---

## ✅ DONE!

Your app will be live at: `https://patient-clustering-dashboard.onrender.com`

---

## 🔑 Important Settings in Render Dashboard:

**Build Command** (already set):
```
pip install --upgrade pip && pip install -r requirements.txt
```

**Start Command** (already set):
```
gunicorn api:app
```

**Environment Variables** (YOU NEED TO ADD):
- `GEMINI_API_KEY` = your_actual_api_key

---

## 💡 Keep Your App Awake (Optional)

Render free tier sleeps after 15 mins of inactivity. To keep it awake:

1. Go to **https://uptimerobot.com** (free)
2. Add new monitor:
   - Type: HTTP(s)
   - URL: `https://patient-clustering-dashboard.onrender.com/health`
   - Interval: 5 minutes
3. Done! Your app stays awake 24/7

---

## 🐛 Troubleshooting

**Build fails?**
- Check logs in Render dashboard
- Make sure all files are pushed to GitHub

**App not loading?**
- Wait 30 seconds (first load after sleep)
- Check if `GEMINI_API_KEY` is set

**Need help?**
- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
