# 🚀 Deploy to Railway (Easiest)

## Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Skin disease classifier API"
git remote add origin https://github.com/YOUR_USERNAME/skin-disease-classifier.git
git push -u origin main
```

## Step 2: Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects FastAPI and deploys!

## Step 3: Get Your Domain
- Railway gives you: `https://your-app-name.railway.app`
- Visit `/docs` for the web interface: `https://your-app-name.railway.app/docs`

## Step 4: Custom Domain (Optional)
- In Railway dashboard → Settings → Domains
- Add your custom domain
- Point DNS to Railway

## 🎯 Your API will be live at:
- **Web Interface**: `https://your-app-name.railway.app/docs`
- **API Endpoint**: `https://your-app-name.railway.app/predict`

## 📱 Test it:
```bash
curl -X POST "https://your-app-name.railway.app/predict" -F "file=@image.jpg"
```
