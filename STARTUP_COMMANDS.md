# Complete Command Sequence for Face Recognition Explainable AI Demo
# ===================================================================

## METHOD 1: Using the Startup Scripts (Recommended)

### For Windows Command Prompt:
```cmd
cd "C:\Users\deepa rajesh\OneDrive\Desktop\faceauth"
start_demo.bat
```

### For PowerShell:
```powershell
cd "C:\Users\deepa rajesh\OneDrive\Desktop\faceauth"
.\start_demo.ps1
```

## METHOD 2: Manual Step-by-Step Commands

### Step 1: Navigate to Project Directory
```cmd
cd "C:\Users\deepa rajesh\OneDrive\Desktop\faceauth"
```

### Step 2: Activate Virtual Environment
```cmd
.\.venv\Scripts\activate
```

### Step 3: Install/Check Dependencies
```cmd
pip install flask streamlit torch torchvision opencv-python pillow plotly matplotlib pandas
```

### Step 4: Run Specific Components

#### Option A: Run Dashboard Only
```cmd
cd ui
python app.py
```
*Access at: http://localhost:5000*

#### Option B: Run Study Server Only
```cmd
cd human_studies
python study_server.py
```
*Access at: http://localhost:5001*

#### Option C: Run Complete Demo
```cmd
python interactive_demo.py --mode full
```
*Dashboard: http://localhost:5000*
*Study Server: http://localhost:5001*

#### Option D: Test Everything
```cmd
python interactive_demo.py --test
```

## METHOD 3: Individual Component Launch

### Dashboard Only:
```cmd
.\.venv\Scripts\activate
python run_dashboard.py
```

### Study Server Only:
```cmd
.\.venv\Scripts\activate
python run_study.py
```

## TROUBLESHOOTING

### If you get import errors:
1. Make sure virtual environment is activated
2. Install missing packages:
   ```cmd
   pip install flask streamlit torch torchvision opencv-python pillow plotly matplotlib pandas
   ```

### If Flask apps don't start:
1. Check that you're in the correct directory
2. Ensure all paths are correct
3. Try running individual components:
   ```cmd
   cd ui
   python -m flask run --port 5000
   ```

### If study server fails:
1. Check the human_studies directory exists
2. Run from the correct directory:
   ```cmd
   cd human_studies
   python study_server.py
   ```

## WHAT EACH COMPONENT DOES

### Main Dashboard (http://localhost:5000)
- Upload and verify face images
- View AI explanations and confidence scores
- Interactive features (prototypes, counterfactuals)
- Accessibility features and high contrast mode

### Study Server (http://localhost:5001)
- Participant recruitment and consent
- Demographics questionnaire
- Explanation evaluation tasks
- Data collection for research

## QUICK START (Recommended)
```cmd
cd "C:\Users\deepa rajesh\OneDrive\Desktop\faceauth"
.\.venv\Scripts\activate
python interactive_demo.py --test
python interactive_demo.py --mode full
```

This will test the system and then launch both components!