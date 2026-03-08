# Deploy DisastraSense (Backend + Frontend) on Render or Vercel

Your project is a **single Flask app**: backend (Python/Flask) and frontend (HTML templates) run together. There is no separate Node/React app.

---

## Option A: Render (recommended for this app)

Render runs your Flask app as a long-lived web service. It supports large dependencies (PyTorch, scikit-learn) and is a good fit for ML apps.

### 1. Push your code to GitHub

Make sure your repo is on GitHub (e.g. `https://github.com/ranjeet229/AI-Disaster-Prediciton`).

### 2. Create a Render account and Web Service

1. Go to [render.com](https://render.com) and sign up (or log in with GitHub).
2. Click **Dashboard** → **New** → **Web Service**.
3. Connect your GitHub repo: **AI-Disaster-Prediciton** (or the repo that contains `AI-Based-Disaster-Prediction-and-Response-System`).
   - If the app lives in a **subfolder** (e.g. `AI-Based-Disaster-Prediction-and-Response-System`), set **Root Directory** to that folder in the next step.

### 3. Configure the Web Service

| Setting | Value |
|--------|--------|
| **Name** | `disastrasense` (or any name) |
| **Region** | Choose one (e.g. Oregon) |
| **Root Directory** | `AI-Based-Disaster-Prediction-and-Response-System` (only if the app is in a subfolder) |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT` |

### 4. Environment variables (optional)

In **Environment** tab add:

- `CHATBASE_API_KEY` – your Chatbase API key (for real chatbot).
- `CHATBASE_CHATBOT_ID` – your Chatbase Chatbot ID.

Leave them empty to use the built-in demo chatbot.

### 5. Deploy

Click **Create Web Service**. Render will build and deploy. When it’s done, you’ll get a URL like `https://disastrasense-xxxx.onrender.com`.

### 6. Models (Flood / Earthquake / Hurricane)

- **Flood** needs `random_forest_model.pkl` and `model_features.pkl` in the repo. If they’re committed, Flood prediction will work.
- **Earthquake** and **Hurricane** need `earthquake_model.pkl` and `hurricane_model.pkl` (and `hurricane_features.pkl`). These are large and not in the repo. Options:
  - **Option 1:** Run `python create_all_models.py` **locally**, then upload the generated `.pkl` files to cloud storage (e.g. S3, Google Drive) and have the app download them at startup (you’d need to add that logic), or
  - **Option 2:** Add a **build step** that runs `python create_all_models.py` during deploy (slower builds, and may hit memory limits on free tier).

For a quick deploy, only Flood may work until you add the other models via one of the options above.

---

## Option B: Vercel (serverless; limitations)

Vercel runs Python as **serverless functions**, not a long-running server. There are limits:

- **Execution time:** 10–60 seconds per request (depending on plan).
- **Payload size:** Large ML models (e.g. PyTorch) can exceed size limits or make cold starts slow.
- Your app loads several `.pkl` models and optionally PyTorch (YOLO); that’s heavy for serverless.

So **Vercel is possible but not ideal** for this app. If you still want to try:

### 1. Install Vercel CLI and add config

In your **project root** (the folder that contains `app.py`):

```bash
npm i -g vercel
```

Create `vercel.json` in the same folder as `app.py`:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

Create `api/index.py` (or keep everything in one app and use a single entry point). For Vercel, you typically expose the Flask app via a serverless handler. Example `api/index.py`:

```python
from app import app
def handler(request, context):
    # Vercel Python uses WSGI; you may need to use a different adapter
    pass
```

Vercel’s Python runtime expects a specific interface; wrapping a full Flask app with multiple routes and large ML models is non-trivial and often hits time/memory limits. So we **recommend Render** for this project.

---

## Summary

| Platform | Best for this project? | Notes |
|----------|------------------------|--------|
| **Render** | Yes | Long-running process, supports PyTorch/sklearn, use `gunicorn` and `PORT`. |
| **Vercel** | No (not ideal) | Serverless, time and size limits; ML models and YOLO are heavy. |

### Quick Render checklist

1. Repo on GitHub.
2. Render → New → Web Service → connect repo.
3. **Root Directory:** `AI-Based-Disaster-Prediction-and-Response-System` if app is in a subfolder.
4. **Build:** `pip install -r requirements.txt`
5. **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Add env vars for Chatbase if you use them.
7. Deploy and open the generated URL.

Your `app.py` is already set to use `PORT` and `0.0.0.0` in production, and the **Procfile** / **render.yaml** in the repo are set for Render.
