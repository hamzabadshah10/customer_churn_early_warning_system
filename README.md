# Customer Churn Early Warning & Business Intelligence Dashboard

## Overview
This project is an advanced machine learning application that predicts customer churn (identifying at-risk accounts for intervention). It uses a **RandomForestClassifier** for prediction and features a fully functional, production-ready Graphical User Interface (GUI) built with **PyQt5**.

### Key Features
- **Custom CSV Ingestion:** Load any custom dataset securely via a native file dialog and process it instantly.
- **Visual Insights:** Embedded **Matplotlib** pie charts that give a high-level breakdown of your retained vs. at-risk customer base.
- **Automated Business Recommendations:** The app automatically extracts the AI model's top feature importances and dynamically recommends intervention strategies (e.g., offering discounts for month-to-month contracts).
- **Real-Time Filtering & Searching:** An interactive search bar that instantly filters thousands of rows to find specific customers or demographics.
- **Multi-Threaded Performance:** The UI remains completely responsive during data loading and model training.
- **Export to CSV:** Easily save the fully analyzed results to a new CSV file for further reporting.

## Screenshots
*(Add 2-3 screenshots of the running PyQt5 application here, showcasing the dark mode, chart, and recommendation panel!)*

## Installation & Setup
1. Create and activate a virtual environment (recommended).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. The dataset will be automatically downloaded via `kagglehub` on the first run if you choose the default analysis.

## Running the App
Execute the main script to launch the GUI:
```bash
python main.py
```
Click **"Run Default Analysis"** to download and process the Telco dataset, or **"Load Custom CSV"** to analyze your own data. 

## How I Used AI
I used AI (specifically Gemini) to assist with:
- Generating the boilerplate project structure.
- Implementing the `RandomForestClassifier` pipeline and properly evaluating it with an 80/20 train-test split (Accuracy, Precision, Recall, and Confusion Matrix).
- Refactoring the GUI into a robust Business Intelligence dashboard with background threading, dark mode styling, matplotlib integration, and real-time data filtering.
