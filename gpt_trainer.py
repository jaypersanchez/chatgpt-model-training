import os
import openai
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QDesktopWidget, QTableView, QPushButton, QTabWidget, QFileDialog
from PyQt5.QtGui import QColor, QPixmap
#import requests
import itertools
import numpy as np
import matplotlib as plt
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt, QRect
import tiktoken
from openai.embeddings_utils import get_embedding
from sklearn.impute import SimpleImputer

openai.organization = "org-PwafGfC5oVQgjaAzYFmgp1ep"
openai.api_key =  "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"

class Satoshi(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GPT Data Modeler')

        # Create tabs
        container = QTabWidget(self)
        tabs = QTabWidget(self)
        training_tab = QWidget()
        data_scaling_tab = QWidget()
        tabs.addTab(training_tab, 'Data Model Training')
        tabs.addTab(data_scaling_tab, "Data Regression Model")
        training_tab.showMaximized()
        data_scaling_tab.showMaximized()
        #tabs.showMaximized() # Unable to maximize each tab

        # Validate data model coins
        self.button_validate_model = QPushButton('Validate Model')
        self.button_validate_model.clicked.connect(self.validate_model)

        # Create data model build file
        self.button_train_gpt = QPushButton('Start GPT Training')
        self.button_train_gpt.clicked.connect(self.train_gpt)
        
        # Exit application
        self.button_exit = QPushButton('Exit')
        self.button_exit.clicked.connect(self.close_application)

        # Create a label to display the message. Used for any error messages
        self.label = QLabel('')
        color = QColor(255,0,0) # red color
        self.label.setStyleSheet("color: {}".format(color.name()))

        # data modelling 
        self.data_textarea = QTextEdit()
        self.data_textarea.setOverwriteMode(True)
        self.data_textarea.toHtml()
        self.data_textarea.setPlaceholderText("Design Data Model in here")

        # status console
        self.console_textarea = QTextEdit()
        self.console_textarea.setOverwriteMode(True)
        self.console_textarea.toHtml()
        self.console_textarea.setPlaceholderText("Debug Output")

        # Create a layout to organize the UI elements
        trainingtab = QVBoxLayout(training_tab) # vertical
        training_tab.setLayout(trainingtab)
        training_tab.setGeometry(QRect(0, 0, self.width(), self.height()))
        datascalingtab = QVBoxLayout(data_scaling_tab)
        data_scaling_tab.setLayout(datascalingtab)
        data_scaling_tab.setGeometry(QRect(0, 0, self.width(), self.height()))
        horizontal_container = QHBoxLayout() # horizontal

        # layout items for Data Model Training
        trainingtab.addWidget(self.button_validate_model)
        trainingtab.addWidget(self.button_train_gpt)
        self.button_validate_model.setDisabled(True)
        self.button_train_gpt.setDisabled(True)
        trainingtab.addLayout(horizontal_container)
        horizontal_container.addWidget(self.data_textarea)
        horizontal_container.addWidget(self.console_textarea)
        trainingtab.setGeometry(QRect(0, 0, self.width(), self.height()))

        # layout items for Data Regression Model
        self.button_import_data_file = QPushButton('Import Data File')
        self.button_import_data_file.clicked.connect(self.importDataFile)
        datascalingtab.addWidget(self.button_import_data_file)

    def importDataFile(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select File", "", "All Files (*);;Python Files (*.py)")
        if fileName:
            print(fileName)
            dataset = pd.read_csv(fileName)
            X = dataset.iloc[:,:-1].values # independent variable
            # dependent variable. the one we want to answer a question
            Y = dataset.iloc[:, -1] # -1 index of last column

            # clean up data like
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, 1:3])
            X[:, 1:3] = imputer.transform(X[:, 1:3])

            # Categorical data
            

            #print
            print(X)
            print(Y)


    # Send prompt to GPT
    def validate_model(self):
        print('Validate Model')
        model = "text-davinci-002"
        '''
        train_data = ["This is an example training sentence.", "Another example sentence for training."]
        validation_data = ["This is an example validation sentence.", "Another example validation sentence."]
        '''
        train_data = "<READ FROM MODEL FILE>" #prompt
        validation_data = "" #complete
        config = {
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "early_stopping": True,
            "validation_split": 0.1
        }
        fine_tune = openai.FineTune.create(model=model, train_data=train_data, validation_data=validation_data, **config)
        # track fine tune status
        run_id = fine_tune["id"]
        status = openai.FineTune.retrieve(run_id)["status"]
        print(f"Fine-tuning run {run_id} is {status}")


    # Train GPT
    def train_gpt(self):
        fine_tuned_model = f"{model}-{run_id}"
        completion = openai.Completion(engine=fine_tuned_model)
        print('Start GPT Training on selected data model')

    # Exit application
    def close_application(self):
        window.close()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Satoshi()
    #screen = QDesktopWidget().screenGeometry()
    window.setGeometry(QDesktopWidget().screenGeometry())
    window.show()
    sys.exit(app.exec_())


