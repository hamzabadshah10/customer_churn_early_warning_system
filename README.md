# Customer Churn Early Warning

## Overview
This project is an advanced machine learning application that predicts customer churn (identifying at-risk accounts for intervention). It uses a **RandomForestClassifier** for prediction and features a fully functional Graphical User Interface (GUI) built with **PyQt5**.

## Screenshots
*(Add 2-3 screenshots of the running PyQt5 application here)*

## Installation & Setup
1. Create and activate a virtual environment (recommended).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. The dataset will be automatically downloaded via `kagglehub` on the first run.

## Running the App
Execute the main script to launch the GUI:
```bash
python main.py
```
Click **"Run Analysis"** to process the data, train the model, and view the highlighted at-risk customers.

## How I Used AI
I used AI (specifically Gemini) to assist with:
- Generating the boilerplate project structure.
- Implementing the `RandomForestClassifier` pipeline (handling categorical encodings with pandas).
- Building the PyQt5 `QTableWidget` interface to visually highlight the predicted churn results.
