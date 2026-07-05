import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, 
                             QTableWidgetItem, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QMessageBox, QLabel,
                             QStatusBar, QFileDialog, QHeaderView, QLineEdit,
                             QAbstractItemView)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from utils import get_clean_data
from model import train_and_predict

# 1. Background Thread for Analysis
class AnalysisThread(QThread):
    finished = pyqtSignal(object, object) # Sends the DataFrame AND top_features dict back
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
                
            predictions, top_features = train_and_predict(df)
            df['Predicted_Churn'] = predictions
            self.finished.emit(df, top_features)
        except Exception as e:
            self.error.emit(f"Failed to process CSV: {str(e)}")

# Canvas class for Matplotlib
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e2e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e2e')
        self.axes.axis('off') # Hide the empty square/ticks before data loads
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Customer Churn Warning - Business Intelligence Dashboard")
        self.setGeometry(100, 100, 1400, 800)
        
        # Apply Dark Mode QSS
        self.apply_dark_theme()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # --- Top Dashboard Layout ---
        self.dashboard_layout = QHBoxLayout()
        
        # Summary Labels
        self.summary_layout = QVBoxLayout()
        self.lbl_total = QLabel("Total Customers: 0")
        self.lbl_risk = QLabel("At-Risk Customers: 0")
        self.lbl_percent = QLabel("Churn Risk %: 0.00%")
        
        for lbl in [self.lbl_total, self.lbl_risk, self.lbl_percent]:
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            self.summary_layout.addWidget(lbl)
        
        # Recommendation Panel
        self.lbl_recommendation = QLabel("<b>Recommendation Framework</b><br>Pending Analysis...")
        self.lbl_recommendation.setFont(QFont("Arial", 11))
        self.lbl_recommendation.setWordWrap(True)
        self.lbl_recommendation.setStyleSheet("background-color: #45475a; border-left: 5px solid #89b4fa; padding: 15px;")
        
        self.dashboard_layout.addLayout(self.summary_layout, stretch=1)
        self.dashboard_layout.addWidget(self.lbl_recommendation, stretch=2)
        
        # Chart Canvas
        self.canvas = MplCanvas(self, width=4, height=3, dpi=100)
        self.dashboard_layout.addWidget(self.canvas, stretch=2)
        
        self.main_layout.addLayout(self.dashboard_layout)
        
        # --- Controls & Search Layout ---
        self.controls_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Run Default Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.controls_layout.addWidget(self.btn_run)
        
        self.btn_load_csv = QPushButton("Load Custom CSV")
        self.btn_load_csv.clicked.connect(self.load_custom_csv)
        self.controls_layout.addWidget(self.btn_load_csv)
        
        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setEnabled(False)
        self.controls_layout.addWidget(self.btn_export)
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search customers by any column...")
        self.search_bar.textChanged.connect(self.filter_table)
        self.search_bar.setStyleSheet("padding: 8px; font-size: 12px; margin-left: 20px;")
        self.controls_layout.addWidget(self.search_bar)
        
        self.main_layout.addLayout(self.controls_layout)
        
        # --- Results Table ---
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.main_layout.addWidget(self.table)
        
        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Ready")
        
        self.df = None
        self.thread = None
        self.top_features = None

    def apply_dark_theme(self):
        dark_stylesheet = """
        QMainWindow { background-color: #1e1e2e; }
        QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', Arial, sans-serif; }
        QLabel { color: #89b4fa; padding: 10px; background-color: #313244; border-radius: 5px; }
        QPushButton { background-color: #89b4fa; color: #1e1e2e; border: none; padding: 10px 20px; font-weight: bold; border-radius: 5px; }
        QPushButton:hover { background-color: #74c7ec; }
        QPushButton:disabled { background-color: #45475a; color: #a6adc8; }
        QLineEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 5px; }
        QTableWidget { background-color: #1e1e2e; alternate-background-color: #2a2b3d; color: #cdd6f4; gridline-color: #313244; border: 1px solid #313244; border-radius: 5px; }
        QTableWidget::item { padding: 5px; }
        QTableWidget::item:selected { background-color: #45475a; color: #cdd6f4; }
        QHeaderView::section { background-color: #313244; color: #89b4fa; font-weight: bold; border: 1px solid #1e1e2e; padding: 5px; }
        QStatusBar { background-color: #11111b; color: #a6adc8; }
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
        self.search_bar.clear()
        
        self.status_bar.showMessage("Processing records... This may take a moment.")
        
        actual_path = csv_path if isinstance(csv_path, str) else None
        self.thread = AnalysisThread(csv_path=actual_path)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.start()

    def on_analysis_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during analysis:\n{error_msg}")
        self.reset_ui()
        self.status_bar.showMessage("Error occurred during processing.")

    def on_analysis_finished(self, df, top_features):
        self.df = df
        self.top_features = top_features
        self.populate_table()
        self.update_dashboard()
        self.update_chart()
        self.update_recommendation()
        self.reset_ui()
        self.btn_export.setEnabled(True)
        self.status_bar.showMessage(f"Successfully processed {len(self.df)} records.")

    def reset_ui(self):
        self.btn_run.setText("Run Default Analysis")
        self.btn_run.setEnabled(True)
        self.btn_load_csv.setEnabled(True)

    def update_dashboard(self):
        if self.df is None or 'Predicted_Churn' not in self.df.columns: return
            
        total = len(self.df)
        at_risk = self.df['Predicted_Churn'].sum()
        percent = (at_risk / total * 100) if total > 0 else 0
        
        self.lbl_total.setText(f"Total Customers: {total}")
        self.lbl_risk.setText(f"At-Risk Customers: {at_risk}")
        self.lbl_percent.setText(f"Churn Risk %: {percent:.2f}%")

    def update_chart(self):
        if self.df is None or 'Predicted_Churn' not in self.df.columns: return
        
        at_risk = self.df['Predicted_Churn'].sum()
        retained = len(self.df) - at_risk
        
        self.canvas.axes.clear()
        
        labels = ['Retained', 'At-Risk']
        sizes = [retained, at_risk]
        colors = ['#89b4fa', '#f38ba8']
        
        wedges, texts, autotexts = self.canvas.axes.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops=dict(color="#cdd6f4", fontweight='bold')
        )
        
        # Make the percentage text inside the pie chart dark so it's clearly readable against the light blue/pink
        for autotext in autotexts:
            autotext.set_color('#1e1e2e')
            autotext.set_fontsize(11)
            
        self.canvas.axes.set_title("Predicted Churn Distribution", color="#cdd6f4", fontweight='bold', pad=10)
        self.canvas.draw()

    def update_recommendation(self):
        if not self.top_features:
            return
            
        top_driver = list(self.top_features.keys())[0]
        
        # Simple programmatic intervention strategy mapping
        rec_text = "Standard retention protocols apply."
        if "Contract_Month-to-month" in top_driver:
            rec_text = "Offer a 10% monthly discount for upgrading to a 1-year or 2-year locked contract."
        elif "InternetService_Fiber optic" in top_driver:
            rec_text = "Reach out regarding Fiber service stability. Offer a free speed upgrade or tech support visit."
        elif "tenure" in top_driver.lower():
            rec_text = "Focus on early onboarding experience. Provide a 3-month introductory credit."
        elif "PaymentMethod_Electronic check" in top_driver:
            rec_text = "Incentivize automatic credit card payments with a $5/mo auto-pay discount."
        elif "TotalCharges" in top_driver or "MonthlyCharges" in top_driver:
            rec_text = "Review billing history. Proactively reach out to offer a personalized, cost-saving bundled plan or a one-time loyalty credit to prevent churn."
            
        self.lbl_recommendation.setText(f"<b>Top Churn Driver Identified:</b><br>{top_driver}<br><br><b>Recommended Intervention:</b><br>{rec_text}")

    def filter_table(self, query):
        if self.df is None: return
        query = query.lower()
        
        self.table.setUpdatesEnabled(False)
        for row in range(self.table.rowCount()):
            match = False
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item and query in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(row, not match)
        self.table.setUpdatesEnabled(True)

    def populate_table(self):
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(self.df.shape[0])
        self.table.setColumnCount(self.df.shape[1])
        self.table.setHorizontalHeaderLabels(self.df.columns)
        
        churn_col_idx = self.df.columns.get_loc('Predicted_Churn')
        warning_color = QColor(138, 30, 40) 
        
        for row in range(self.df.shape[0]):
            is_churn = self.df.iat[row, churn_col_idx] == 1
            bg_color = warning_color if is_churn else None
            
            for col in range(self.df.shape[1]):
                val = self.df.iat[row, col]
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if bg_color:
                    item.setBackground(bg_color)
                self.table.setItem(row, col, item)
                
        self.table.setUpdatesEnabled(True)

    def export_csv(self):
        if self.df is None: return
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
