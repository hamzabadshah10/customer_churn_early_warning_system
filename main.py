import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QColor

from utils import get_clean_data
from model import train_and_predict

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Customer Churn Warning")
        self.setGeometry(100, 100, 1000, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.btn_run)
        
        self.table = QTableWidget()
        self.layout.addWidget(self.table)
        
        self.df = None

    def run_analysis(self):
        # Update button text to show progress
        self.btn_run.setText("Analyzing...")
        self.btn_run.setEnabled(False)
        QApplication.processEvents() # Force UI update
        
        try:
            # 1. Clean data
            self.df = get_clean_data()
            
            # 2. Get predictions
            predictions = train_and_predict(self.df)
            
            # Add predictions to dataframe for display
            self.df['Predicted_Churn'] = predictions
            
            # 3. Update UI
            self.populate_table()
        finally:
            self.btn_run.setText("Run Analysis")
            self.btn_run.setEnabled(True)

    def populate_table(self):
        self.table.setRowCount(self.df.shape[0])
        self.table.setColumnCount(self.df.shape[1])
        self.table.setHorizontalHeaderLabels(self.df.columns)
        
        # Populate table items
        for row in range(self.df.shape[0]):
            for col in range(self.df.shape[1]):
                val = self.df.iat[row, col]
                item = QTableWidgetItem(str(val))
                
                # Highlight if predicted churn is 1
                if self.df['Predicted_Churn'].iloc[row] == 1:
                    item.setBackground(QColor(255, 200, 200)) # Light red for at-risk customers
                    
                self.table.setItem(row, col, item)

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
