# AI-Based-Disaster-Prediction-and-Response-System-

Overview

The AI-Based Disaster Prediction and Response System is designed to leverage the power of Artificial Intelligence and Machine Learning to predict natural disasters and enhance response efforts. This project focuses on forecasting floods, earthquakes, and hurricanes, and provides tools to identify and assist victims post-disaster.

#Features
Predictive Models for Natural Disasters:

1.Flood Prediction Model: Analyzes historical weather patterns, river levels, and precipitation data to forecast potential flooding events.

2.Earthquake Prediction Model: Utilizes seismic activity and geological data to predict earthquake occurrences and their potential impacts.

3.Hurricane Prediction Model: Uses atmospheric data, ocean temperatures, and historical storm patterns to predict hurricane formations and trajectories.

Chatbase API Integration:

Integrated with Chatbase API to offer an interactive chat-based interface. Users can receive real-time responses and information related to flood scenarios and safety measures.

YOLOv5 Model for Post-Disaster Response:

Deployed YOLOv5 (You Only Look Once) object detection model to analyze post-disaster imagery and videos. This model helps in locating and identifying victims, aiding in efficient rescue operations.

Input video- https://drive.google.com/file/d/1JavezBw-sL8aN0rA28tajhg9rKx9-IJC/view

Output video- https://drive.google.com/file/d/1m8MW_zcCqczPru_CO85Y5WTe3nfdc9zE/view?pli=1


Technology Stack

Machine Learning Models: Developed using Python, pandas, numpy, and scikit-learn.

Chatbase API: Provides a responsive chat interface for user interaction.

YOLOv5: Utilized for real-time object detection and victim identification in disaster-stricken areas.

Html,css,js and NodeJs: For creating Frontend and Backend Website Integration with AI/ML.

How It Works

Prediction Phase:

1.Data Collection: Gathers relevant data from various sources, including weather stations, seismic sensors, and oceanographic data.

2.Model Training: Uses historical data to train machine learning models capable of predicting floods, earthquakes, and hurricanes.

3.Prediction: The models generate forecasts based on current data inputs, providing early warnings for potential disasters.

---

## How to Run This Project (Quick Start)

**DisastraSense** runs as a single-page app with one navbar and a dropdown to switch prediction type.

**1. Open a terminal** (PowerShell or Command Prompt).

**2. Go to the project folder:**
```bash
cd "D:\11-E-DRIVE\WEB_DEV\capstonedisaster\AI-Based-Disaster-Prediction-and-Response-System"
```
*(If your project is elsewhere, use that path instead.)*

**3. (First time only)** Install dependencies:
```bash
pip install -r requirements.txt
```

**4. (First time only)** Create all prediction models (Flood, Earthquake, Hurricane):
```bash
python create_all_models.py
```

**5. Start the application:**
```bash
python app.py
```

**6. Open your browser** and go to: **http://127.0.0.1:5000**

**7. To stop the app:** Press `Ctrl+C` in the terminal.

**8. In the app:** You’ll see the **home page** (Welcome to Disaster Prediction & Management). Click **Get started** to open the main app. Use the **navbar** – left: **DisastraSense**; right: **theme toggle (Light/Dark)** and **dropdown** to switch between Flood, Earthquake, Hurricane, Chatbase Chatbot, and YOLO. Default theme is **light**.

When you run `python app.py`, the terminal prints an **Open in browser** link (e.g. `http://127.0.0.1:5000`). Click it in the terminal to open the app.

*(Terminal warnings from the dev server are suppressed.)*

---

## Chatbase Chatbot – How to create API key and Chatbot ID

To use the **real Chatbase API** instead of demo replies, you need an **API key** and a **Chatbot ID**. You can use the **free plan**.

**Note:** Chatbase may require a **paid plan** (Standard or Pro) to unlock API access. If you see “Upgrade your plan to use this feature” when creating an API key, you have two options: (1) use the **built-in demo chatbot** in this app (no key needed – see below), or (2) subscribe to a Chatbase plan if you want their full AI.

Here’s how to get your API key and Chatbot ID:

### Step 1: Sign up and open the dashboard

1. Go to **[https://www.chatbase.co](https://www.chatbase.co)**.
2. Click **Sign up** (or **Log in** if you already have an account).
3. After login, you’ll see the **Dashboard** ([https://www.chatbase.co/dashboard](https://www.chatbase.co/dashboard)).

### Step 2: Create / get your API key

1. In the dashboard, open **Workspace Settings** (gear or profile menu).
2. Go to **API Keys**.
3. Click **Create API Key**.
4. Copy the generated key and store it somewhere safe (you won’t see it again in full).  
   → This is your **CHATBASE_API_KEY**.

### Step 3: Create or select a chatbot and get Chatbot ID

1. In the dashboard, click **New chat** or **Create chatbot** to create a new AI agent, or select an existing one.
2. Give it a name (e.g. “Disaster Safety Bot”) and add any training data or instructions you want.
3. Open that chatbot’s **Settings** (gear icon or **Settings** tab).
4. Go to **General**.
5. Find **Chatbot ID** (a long UUID like `abc12345-...`). Copy it.  
   → This is your **CHATBASE_CHATBOT_ID**.

### Step 4: Use them in this app

**How to set the API key and Chatbot ID**
   - **Option A (environment variables)** – before starting the app:
     - **Windows (PowerShell):**  
       `$env:CHATBASE_API_KEY="your-api-key"; $env:CHATBASE_CHATBOT_ID="your-chatbot-id"`
     - **Windows (CMD):**  
       `set CHATBASE_API_KEY=your-api-key` and `set CHATBASE_CHATBOT_ID=your-chatbot-id`
     - **Linux/macOS:**  
       `export CHATBASE_API_KEY=your-api-key` and `export CHATBASE_CHATBOT_ID=your-chatbot-id`
   - **Option B (.env file)** – create a `.env` file in the project folder with:
     ```
     CHATBASE_API_KEY=your-api-key
     CHATBASE_CHATBOT_ID=your-chatbot-id
     ```
     The app loads `.env` automatically (uses `python-dotenv`; run `pip install -r requirements.txt`).

**Restart the app** after setting the variables. The chatbot will then use the real Chatbase API; without them, you get built-in demo replies.

---

## YOLOv5 Victim Detection

The app runs **YOLOv5** (via PyTorch Hub) for person detection in images. It needs **PyTorch** (`torch`, `torchvision` in `requirements.txt`). The first time you use the “Analyze image” button, the app will download the `yolov5s` model (~15 MB). Upload an image in the **YOLO Victim Detection** panel and click **Analyze image** to get the person count and an annotated image. For a smaller install (CPU-only), you can use:  
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

## What’s in this repo (DisastraSense single-page app)

| Feature | Status | Notes |
|--------|--------|--------|
| **Flood Prediction** | ✅ Working | Form in 3 columns → Flood risk Yes/No. |
| **Earthquake Prediction** | ✅ Working | Latitude, Longitude → predicted magnitude. Uses `earthquake (1).csv`. |
| **Hurricane Prediction** | ✅ Working | Lat, Lon, wind fields, date → predicted max wind (knots). Uses `atlantic.csv`. |
| **Chatbase Chatbot** | ✅ Demo | In-app chat; stub replies (no external API). Add your API key in backend for full Chatbase. |
| **YOLO Victim Detection** | ✅ Working | Upload an image → YOLOv5 detects persons (victims). Shows count and annotated image. Uses PyTorch Hub (`yolov5s`). |

---

This repository contains large files tracked by Git LFS. To properly download and set up the project, follow these steps:

Prerequisites
Git

Git LFS

Python

pip

Node.js

npm

Installation

Install Git LFS If you don't have Git LFS installed, download and install it from the Git LFS installation page.

Clone the repository

git clone  https://github.com/t-abs/AI-Based-Disaster-Prediction-and-Response-System

cd Predicaster

Set up the Predicaster backend

cd Predicater/Predicaster

pip install -r requirements.txt

python app.py

Set up the Client (in a new terminal window)

cd ../client

npm i

npm start


#Response Phase:

1.Chat Interface: Users interact with the system through a chat interface powered by Chatbase API to get real-time updates and safety advice.

2.Victim Detection: Post-disaster, the YOLOv5 model processes images and videos to identify and locate victims, streamlining rescue operations.

