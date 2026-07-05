import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, 
                             QTableWidgetItem, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QMessageBox, QLabel,
                             QStatusBar, QFileDialog, QHeaderView)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from utils import get_clean_data
from model import train_and_predict

# 1. Background Thread for Analysis
class AnalysisThread(QThread):
    finished = pyqtSignal(object) # Sends the DataFrame back
    error = pyqtSignal(str)       # Sends error message

    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            if self.csv_path:
                df = pd.read_csv(self.csv_path)
                if 'Churn' not in df.columns:
                    raise ValueError("The selected CSV is missing the required 'Churn' column.")
                
                if 'customerID' in df.columns:
                    df = df.drop(columns=['customerID'])
                if 'TotalCharges' in df.columns:
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
                
                if df['Churn'].dtype == 'O':
                    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
                
                df = df.dropna()
            else:
                df = get_clean_data()
                
            predictions = train_and_predict(df)
            df['Predicted_Churn'] = predictions
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(f"Failed to process CSV: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Customer Churn Warning - Professional Dashboard")
        self.setGeometry(100, 100, 1200, 700)
        
        # Apply Dark Mode QSS
        self.apply_dark_theme()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        # --- Summary Dashboard ---
        self.dashboard_layout = QHBoxLayout()
        
        self.lbl_total = QLabel("Total Customers: 0")
        self.lbl_risk = QLabel("At-Risk Customers: 0")
        self.lbl_percent = QLabel("Churn Risk %: 0.00%")
        
        # Styling the labels
        for lbl in [self.lbl_total, self.lbl_risk, self.lbl_percent]:
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            self.dashboard_layout.addWidget(lbl)
            
        self.layout.addLayout(self.dashboard_layout)
        
        # --- Controls Layout ---
        self.controls_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Run Default Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.controls_layout.addWidget(self.btn_run)
        
        self.btn_load_csv = QPushButton("Load Custom CSV")
        self.btn_load_csv.clicked.connect(self.load_custom_csv)
        self.controls_layout.addWidget(self.btn_load_csv)
        
        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setEnabled(False) # Disabled until data is ready
        self.controls_layout.addWidget(self.btn_export)
        
        self.layout.addLayout(self.controls_layout)
        
        # --- Results Table ---
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        # Make the table fill available space gracefully
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.layout.addWidget(self.table)
        
        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Ready")
        
        self.df = None
        self.thread = None

    def apply_dark_theme(self):
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e2e;
        }
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QLabel {
            color: #89b4fa;
            padding: 10px;
            background-color: #313244;
            border-radius: 5px;
        }
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #74c7ec;
        }
        QPushButton:disabled {
            background-color: #45475a;
            color: #a6adc8;
        }
        QTableWidget {
            background-color: #1e1e2e;
            alternate-background-color: #2a2b3d;
            color: #cdd6f4;
            gridline-color: #313244;
            border: 1px solid #313244;
            border-radius: 5px;
        }
        QTableWidget::item {
            padding: 5px;
        }
        QTableWidget::item:selected {
            background-color: #45475a;
            color: #cdd6f4;
        }
        QHeaderView::section {
            background-color: #313244;
            color: #89b4fa;
            font-weight: bold;
            border: 1px solid #1e1e2e;
            padding: 5px;
        }
        QStatusBar {
            background-color: #11111b;
            color: #a6adc8;
        }
        """
        self.setStyleSheet(dark_stylesheet)

    def load_custom_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.run_analysis(csv_path=file_path)

    def run_analysis(self, csv_path=None):
        self.btn_run.setText("Analyzing...")
        self.btn_run.setEnabled(False)
        self.btn_load_csv.setEnabled(False)
        self.btn_export.setEnabled(False)
        
        self.status_bar.showMessage("Processing records... This may take a moment.")
        
        # Start background thread (protecting against PyQt signal passing a boolean)
        actual_path = csv_path if isinstance(csv_path, str) else None
        self.thread = AnalysisThread(csv_path=actual_path)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.start()

    def on_analysis_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during analysis:\n{error_msg}")
        self.reset_ui()
        self.status_bar.showMessage("Error occurred during processing.")

    def on_analysis_finished(self, df):
        self.df = df
        self.populate_table()
        self.update_dashboard()
        self.reset_ui()
        self.btn_export.setEnabled(True)
        self.status_bar.showMessage(f"Successfully processed {len(self.df)} records.")

    def reset_ui(self):
        self.btn_run.setText("Run Default Analysis")
        self.btn_run.setEnabled(True)
        self.btn_load_csv.setEnabled(True)

    def update_dashboard(self):
        if self.df is None or 'Predicted_Churn' not in self.df.columns:
            return
            
        total = len(self.df)
        at_risk = self.df['Predicted_Churn'].sum()
        percent = (at_risk / total * 100) if total > 0 else 0
        
        self.lbl_total.setText(f"Total Customers: {total}")
        self.lbl_risk.setText(f"At-Risk Customers: {at_risk}")
        self.lbl_percent.setText(f"Churn Risk %: {percent:.2f}%")

    def populate_table(self):
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        
        self.table.setRowCount(self.df.shape[0])
        self.table.setColumnCount(self.df.shape[1])
        self.table.setHorizontalHeaderLabels(self.df.columns)
        
        churn_col_idx = self.df.columns.get_loc('Predicted_Churn')
        
        # Darker red for dark mode warning
        warning_color = QColor(138, 30, 40) 
        
        for row in range(self.df.shape[0]):
            is_churn = self.df.iat[row, churn_col_idx] == 1
            bg_color = warning_color if is_churn else None
            
            for col in range(self.df.shape[1]):
                val = self.df.iat[row, col]
                item = QTableWidgetItem(str(val))
                
                # Make items read-only
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                
                if bg_color:
                    item.setBackground(bg_color)
                    
                self.table.setItem(row, col, item)
                
        self.table.setUpdatesEnabled(True)

    def export_csv(self):
        if self.df is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "churn_results.csv", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.status_bar.showMessage(f"Exporting to {file_path}...")
                self.df.to_csv(file_path, index=False)
                self.status_bar.showMessage(f"Successfully exported to {file_path}")
                QMessageBox.information(self, "Success", "Results exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export file:\n{str(e)}")
                self.status_bar.showMessage("Export failed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
