import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTabWidget,
    QComboBox, QLineEdit, QFileDialog, QMessageBox, QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt

class MainWindow(QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.add_first_tab()
        self.add_second_tab()
        
        self.setWindowTitle("PyQt5 Example")
        self.show()

    def add_first_tab(self):
        """Create the first tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        label = QLabel("Select an option:")
        layout.addWidget(label)
        
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(self.combo_box)
        
        self.button = QPushButton("Choose File")
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)
        
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        layout.addWidget(self.file_path)
        
        self.tabs.addTab(tab, "Tab 1")

    def add_second_tab(self):
        """Create the second tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        layout.addWidget(self.slider)
        
        self.label = QLabel("Slider Value: 50")
        layout.addWidget(self.label)
        
        self.slider.valueChanged.connect(self.update_label)
        
        self.tabs.addTab(tab, "Tab 2")

    def open_file_dialog(self):
        """Open a file dialog to select a file."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_name:
            self.file_path.setText(file_name)

    def update_label(self, value):
        """Update the label with the slider value."""
        self.label.setText(f"Slider Value: {value}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
