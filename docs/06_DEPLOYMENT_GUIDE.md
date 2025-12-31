# 🚀 Deployment Guide - Production Deployment

## 🎯 Deployment Objectives

1. **Deploy to cloud** untuk accessibility 24/7
2. **Ensure scalability** untuk multiple concurrent users
3. **Secure model artifacts** dan data privacy
4. **Monitor performance** dan error tracking
5. **Enable easy updates** untuk model retraining

---

## 📋 Pre-Deployment Checklist

### Required Files

```
mbg-fraud-detection/
├── app.py                    ✅ Main dashboard
├── autoencoder.h5            ✅ Trained model
├── scaler.pkl                ✅ Fitted preprocessor
├── mbg_synthetic.csv         ✅ Demo data
├── requirements.txt          ✅ Dependencies
├── .streamlit/
│   └── config.toml          ✅ Streamlit config
└── README.md                ✅ Documentation
```

### Verify Local Functionality

```bash
# Test locally before deployment
streamlit run app.py

# Expected: Dashboard loads without errors
# Test upload, threshold adjustment, export
```

---

## 🌐 Deployment Option 1: Streamlit Cloud (Recommended)

### Why Streamlit Cloud?
- ✅ **Free tier available** (Community plan)
- ✅ **Zero infrastructure management**
- ✅ **Automatic SSL certificate**
- ✅ **GitHub integration** untuk continuous deployment
- ✅ **Built-in secrets management**

---

### Step 1: Prepare Repository

#### Create `.streamlit/config.toml`

```toml
// filepath: c:\Users\ROBBY_DATA\Repos\mbg\mbg-fraud-detection\.streamlit\config.toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

---

#### Create `runtime.txt` (Optional - Specify Python Version)

```text
// filepath: c:\Users\ROBBY_DATA\Repos\mbg\mbg-fraud-detection\runtime.txt
python-3.11
```

**Note**: Streamlit Cloud supports Python 3.8-3.11 (as of Dec 2024).

---

### Step 2: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - MBG Fraud Detection"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/mbg-fraud-detection.git

# Push to GitHub
git push -u origin main
```

---

### Step 3: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** dengan GitHub account
3. **Click**: "New app"
4. **Select**:
   - Repository: `YOUR_USERNAME/mbg-fraud-detection`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click**: "Deploy!"

**Deployment time**: 2-5 minutes

**Your app will be live at**: `https://YOUR_USERNAME-mbg-fraud-detection-app-xxxxx.streamlit.app`

---

### Step 4: Configure Secrets (If Needed)

If you have API keys or sensitive data:

1. Go to app settings (⚙️ icon)
2. Click "Secrets"
3. Add secrets in TOML format:

```toml
[database]
host = "your-db-host"
username = "your-username"
password = "your-password"
```

Access in code:
```python
import streamlit as st

db_host = st.secrets["database"]["host"]
```

---

### Step 5: Monitor & Update

#### View Logs
- Click "Manage app" → "Logs"
- See real-time error messages & user activity

#### Update App
```bash
# Make changes to code
git add .
git commit -m "Update: improved validation"
git push

# Streamlit Cloud auto-redeploys! 🎉
```

---

## 🐳 Deployment Option 2: Docker Container

### Why Docker?
- ✅ **Consistent environment** across machines
- ✅ **Easy scaling** dengan orchestration (Kubernetes)
- ✅ **Version control** untuk entire application stack
- ✅ **Deploy anywhere** (AWS, Azure, GCP, on-premise)

---

### Step 1: Create Dockerfile

```dockerfile
// filepath: c:\Users\ROBBY_DATA\Repos\mbg\mbg-fraud-detection\Dockerfile
# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### Step 2: Create .dockerignore

```text
// filepath: c:\Users\ROBBY_DATA\Repos\mbg\mbg-fraud-detection\.dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
.vscode
.idea
*.md
docs/
*.png
*.csv
!mbg_synthetic.csv
```

---

### Step 3: Build Docker Image

```bash
# Build image
docker build -t mbg-fraud-detection:v1.0 .

# Expected output:
# Successfully built abc123def456
# Successfully tagged mbg-fraud-detection:v1.0
```

---

### Step 4: Run Container Locally (Test)

```bash
# Run container
docker run -p 8501:8501 mbg-fraud-detection:v1.0

# Access at: http://localhost:8501
```

**Test**:
- ✅ Dashboard loads
- ✅ Upload CSV works
- ✅ Model prediction works
- ✅ Export functionality works

---

### Step 5: Push to Container Registry

#### Docker Hub
```bash
# Login
docker login

# Tag image
docker tag mbg-fraud-detection:v1.0 YOUR_USERNAME/mbg-fraud-detection:v1.0

# Push
docker push YOUR_USERNAME/mbg-fraud-detection:v1.0
```

#### AWS ECR
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag mbg-fraud-detection:v1.0 YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mbg-fraud-detection:v1.0

# Push
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mbg-fraud-detection:v1.0
```

---

## ☁️ Deployment Option 3: AWS EC2

### Step 1: Launch EC2 Instance

```bash
# Instance type: t3.medium (2 vCPU, 4GB RAM)
# OS: Ubuntu 22.04 LTS
# Security Group: Allow port 8501 (HTTP)
```

---

### Step 2: Connect & Setup

```bash
# SSH to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu
```

---

### Step 3: Deploy Container

```bash
# Pull Docker image
docker pull YOUR_USERNAME/mbg-fraud-detection:v1.0

# Run container with restart policy
docker run -d \
  --name mbg-fraud \
  --restart unless-stopped \
  -p 8501:8501 \
  YOUR_USERNAME/mbg-fraud-detection:v1.0
```

---

### Step 4: Setup Nginx Reverse Proxy (Optional)

```bash
# Install Nginx
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/mbg-fraud
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/mbg-fraud /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

### Step 5: SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

**Your app is now live at**: `https://your-domain.com` 🎉

---

## 📊 Deployment Option 4: Heroku

### Step 1: Create Heroku App

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create mbg-fraud-detection
```

---

### Step 2: Create Procfile

```text
// filepath: c:\Users\ROBBY_DATA\Repos\mbg\mbg-fraud-detection\Procfile
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

### Step 3: Deploy

```bash
# Add Heroku remote
heroku git:remote -a mbg-fraud-detection

# Deploy
git push heroku main

# Open app
heroku open
```

**Your app is live at**: `https://mbg-fraud-detection.herokuapp.com`

---

## 🔒 Security Considerations

### 1. Environment Variables

**Never commit sensitive data!**

```python
# Use environment variables
import os

DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
```

**Set in Streamlit Cloud**:
- App Settings → Secrets

**Set in Docker**:
```bash
docker run -e DATABASE_URL="your-url" -e API_KEY="your-key" ...
```

---

### 2. File Upload Security

```python
# Limit file size (already implemented in app.py)
if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB
    st.error("File too large!")
```

---

### 3. Input Validation

```python
# Validate CSV content (already implemented)
def validate_dataframe(df):
    # 9-point validation checklist
    ...
```

---

### 4. Rate Limiting

**Option A: Streamlit Cloud** (built-in rate limiting)

**Option B: Nginx** (for self-hosted):
```nginx
limit_req_zone $binary_remote_addr zone=mylimit:10m rate=10r/s;

server {
    location / {
        limit_req zone=mylimit burst=20;
        proxy_pass http://localhost:8501;
    }
}
```

---

## 📈 Monitoring & Logging

### Streamlit Cloud

**Built-in Monitoring**:
- View logs in real-time
- See user activity
- Track errors

---

### Self-Hosted (Docker/EC2)

#### 1. Application Logs

```python
# Add logging to app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log events
logger.info(f"User uploaded file: {uploaded_file.name}")
logger.error(f"Error loading model: {str(e)}")
```

---

#### 2. System Monitoring

```bash
# Install monitoring tools
sudo apt install htop nethogs -y

# View resource usage
htop

# View network usage
sudo nethogs
```

---

#### 3. Docker Logs

```bash
# View container logs
docker logs mbg-fraud

# Follow logs in real-time
docker logs -f mbg-fraud

# View last 100 lines
docker logs --tail 100 mbg-fraud
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

### Create Workflow

```yaml
// filepath: c:\Users\ROBBY_DATA\Repos\mbg\mbg-fraud-detection\.github\workflows\deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          python -m pytest tests/
      
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Streamlit Cloud redeploy
        run: echo "Streamlit Cloud auto-deploys on push"
```

---

## 🎯 Deployment Comparison

| Feature | Streamlit Cloud | Docker (Self-hosted) | Heroku | AWS EC2 |
|---------|----------------|----------------------|--------|---------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Cost** | Free tier available | Server cost | $7/month (hobby) | $10-50/month |
| **Scalability** | Medium | High | Medium | High |
| **Control** | Low | High | Medium | High |
| **Maintenance** | Zero | Medium | Low | High |
| **SSL** | Automatic | Manual | Automatic | Manual |
| **Custom Domain** | ❌ Free tier | ✅ | ✅ | ✅ |
| **Best For** | MVP, demos | Production | Prototypes | Enterprise |

---

## 🚀 Post-Deployment Checklist

### Functionality
- [ ] Dashboard loads successfully
- [ ] Demo data works
- [ ] CSV upload works (valid & invalid files)
- [ ] Threshold slider updates predictions
- [ ] Export CSV downloads correctly
- [ ] Metrics display accurately (if ground truth available)

### Performance
- [ ] Load time < 5 seconds
- [ ] Prediction time < 2 seconds (1000 rows)
- [ ] No memory leaks (monitor over 24 hours)

### Security
- [ ] No sensitive data in code
- [ ] Environment variables configured
- [ ] SSL enabled (for production)
- [ ] File upload size limited

### Monitoring
- [ ] Logging configured
- [ ] Error tracking enabled
- [ ] Resource monitoring setup

---

## 📚 Troubleshooting

### Issue 1: App Crashes on Startup

**Symptom**: "ModuleNotFoundError: No module named 'X'"

**Solution**:
```bash
# Check requirements.txt includes all dependencies
pip freeze > requirements.txt

# Rebuild/redeploy
```

---

### Issue 2: Model Not Loading

**Symptom**: "FileNotFoundError: autoencoder.h5 not found"

**Solution**:
```bash
# Ensure model files are committed to Git
git add autoencoder.h5 scaler.pkl
git commit -m "Add model artifacts"
git push
```

---

### Issue 3: Slow Performance

**Solution**:
```python
# Add caching (already implemented in app.py)
@st.cache_resource
def load_model_cached():
    ...

# Use smaller demo dataset
if len(df) > 1000:
    df = df.sample(1000)
```

---

### Issue 4: Memory Issues

**Solution**:
```bash
# Increase Docker memory limit
docker run -m 2g -p 8501:8501 mbg-fraud-detection

# Or upgrade Streamlit Cloud plan
```

---

## 🎯 Recommended Deployment Strategy

### For MVP/Testing
✅ **Streamlit Cloud** (Free, easy, fast)

### For Production
✅ **Docker on AWS EC2/ECS** (Control, scalability, monitoring)

### For Enterprise
✅ **Kubernetes cluster** (High availability, auto-scaling, load balancing)

---

**Document Version**: 1.0  
**Last Updated**: December 31, 2025 
**Next**: [07_USER_MANUAL.md](07_USER_MANUAL.md)