# Quick Start Guide

## 🚀 Getting Started in 5 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset
```bash
python data_generator.py
```
This creates `data/dataset.csv` with 5,000 student records.

### Step 3: Run EDA (Optional but Recommended)
```bash
python eda/run_eda.py
```
This generates analysis plots and reports in `eda/plots/` and `eda/eda_report.md`.

### Step 4: Train Model in Google Colab

1. **Upload dataset to Google Drive:**
   - Upload `data/dataset.csv` to `MyDrive/AI_Project/data/`

2. **Open Colab notebook:**
   - Open `colab/train.ipynb` in Google Colab
   - Or create new notebook and copy the code

3. **Run all cells:**
   - The notebook will train XGBoost and RandomForest models
   - Models will be saved to Google Drive

4. **Download model:**
   - Download `dropout_model.pkl` from `MyDrive/AI_Project/models/`
   - Place it in your local `models/` folder

### Step 5: Run Application
```bash
python app/main.py
```

## 📝 First Time Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Database initialized (`python database/init_db.py`)
- [ ] Dataset generated (`python data_generator.py`)
- [ ] Model trained in Colab and downloaded to `models/`
- [ ] Webcam connected and working

## 🎯 Common First Tasks

1. **Register a Student:**
   - Click "Register New Student" in the app
   - Enter student information
   - Capture face (look at camera)

2. **Mark Attendance:**
   - Click "Scan Face & Identify"
   - Look at camera
   - Click "Mark Attendance"

3. **Check Prediction:**
   - After identifying a student
   - Click "Check Dropout Prediction"
   - View risk assessment

## ⚠️ Troubleshooting

**"Model not found" error:**
- Make sure you've trained the model in Colab
- Download `dropout_model.pkl` to `models/` folder

**"Camera not working":**
- Check webcam permissions
- Try different camera index in code (0, 1, 2...)

**"Face not recognized":**
- Ensure good lighting
- Face should be clearly visible
- Re-register with better quality

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check `eda/eda_report.md` for data insights
- Explore the code in `app/` folder
- Customize models in `colab/train.ipynb`

---

**Need Help?** Check the troubleshooting section in README.md


