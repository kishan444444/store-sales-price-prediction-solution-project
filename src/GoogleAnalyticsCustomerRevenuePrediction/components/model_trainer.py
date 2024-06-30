import os
import sys
import pandas as pd
import numpy as np
from src.GoogleAnalyticsCustomerRevenuePrediction.logger import logging
from src.GoogleAnalyticsCustomerRevenuePrediction.exception import customexception
from src.GoogleAnalyticsCustomerRevenuePrediction.utils.utils import save_object
from src.GoogleAnalyticsCustomerRevenuePrediction.utils.utils import evaluate_model
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerconfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting dependent and independented variable from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "random forest":RandomForestRegressor()
            }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("/n===================================================================================/n")
            logging.info(f"model_report,{ model_report}")
            
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            print(f'Best Model Found , Model Name : {best_model_name} ,  r2_score  : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} ,  r2_score : {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )
            
            
        except Exception as e:
            logging.info("Exception occured at model trainer")
            raise customexception(sys,e)
    
