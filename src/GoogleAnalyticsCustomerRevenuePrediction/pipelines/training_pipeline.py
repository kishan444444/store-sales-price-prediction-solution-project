from src.GoogleAnalyticsCustomerRevenuePrediction.components.data_ingestion import Dataingestion
from src.GoogleAnalyticsCustomerRevenuePrediction.components.data_transformation import DataTransformation
from src.GoogleAnalyticsCustomerRevenuePrediction.components.model_trainer import ModelTrainer
import pandas as pd


data_ingestion=Dataingestion()

train_path,test_path=data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation()

train_array,test_array=data_transformation.initiated_data_transformation(train_path,test_path)

model_trainer=ModelTrainer()

model_trainer.initiate_model_training(train_array,test_array)

