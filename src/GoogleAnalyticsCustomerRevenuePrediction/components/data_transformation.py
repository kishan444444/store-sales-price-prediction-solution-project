import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.GoogleAnalyticsCustomerRevenuePrediction.exception import customexception
from src.GoogleAnalyticsCustomerRevenuePrediction.logger import logging

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.GoogleAnalyticsCustomerRevenuePrediction.utils.utils import save_object




@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=Data_transformation_config()
       
    def get_data_transformation(self):
        
        try:
            #Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type',
                             'Outlet_Type']
            numerical_cols=['Item_Identifier', 'Item_Weight', 'Item_Visibility', 'Item_MRP']
            
            # Define the custom ranking for each ordinal variable
            
            Item_Fat_Content_map=["Low Fat","Regular","LF","reg","low fat"]
            Item_Type_map=["Fruits and Vegetables","Snack Foods","Household","Frozen Foods","Dairy","Canned","Baking Goods","Health and Hygiene",
                           "Soft Drinks","Meat","Breads","Hard Drinks","Others","Starchy Foods","Breakfast","Seafood"]
            Outlet_Size_map=["Medium","medium","Small","High"]
            Outlet_Location_Type_map=["Tier 3","Tier 2","Tier 1"]
            Outlet_Type_map=["Supermarket Type1","Grocery Store","Supermarket Type3","Supermarket Type2"]
            
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Item_Fat_Content_map,Item_Type_map,Outlet_Size_map,Outlet_Location_Type_map,Outlet_Type_map])),
                ('scaler',StandardScaler())
                ]

            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
           
        except Exception as e:
            logging.info("Exception occured in initiating data transformation")
            raise customexception(e,sys)
        
    def initiated_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data completed")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj=self.get_data_transformation()
            
            train_df = pd.DataFrame(train_df)
            test_df=pd.DataFrame(train_df)
            
            mean_value= train_df.loc[:, 'Item_Weight'].mean()
            train_df['Item_Weight'].fillna(value=mean_value, inplace=True) 
            
            mean_value= test_df.loc[:, 'Item_Weight'].mean()
            test_df['Item_Weight'].fillna(value=mean_value, inplace=True) 
            
            
            import re
            # Function to extract numeric part from a string
            def extract_numeric(value):
                numeric_part = re.findall(r'\d+', value)  # Find all numeric parts
                return int(numeric_part[0]) if numeric_part else None  # Convert to int if found
            # Apply the function to extract numeric part and convert to int
                
            train_df['Item_Identifier'] = train_df['Item_Identifier'].apply(extract_numeric)
            test_df['Item_Identifier'] = test_df['Item_Identifier'].apply(extract_numeric)
            
            input_feature_train_df = train_df.drop("Outlet_Establishment_Year",axis=1)
            input_feature_train_df = train_df.drop("Outlet_Identifier",axis=1)
            input_feature_train_df = train_df.drop("Item_Outlet_Sales",axis=1)
            target_feature_train_df=train_df["Item_Outlet_Sales"]
            
            
            input_feature_test_df = test_df.drop("Outlet_Establishment_Year",axis=1)
            input_feature_test_df = test_df.drop("Outlet_Identifier",axis=1)
            input_feature_test_df = test_df.drop("Item_Outlet_Sales",axis=1)
            target_feature_test_df=test_df["Item_Outlet_Sales"]
            
            
            
            input_feature_train_arr=preprocessing_obj.fit_transform( input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
          
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                        )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr,
               
                )
        
        
        except Exception as e:
            logging.info("Exception occured in initiating data transformation")
            raise customexception(e,sys)
        
        
    
        
        

            