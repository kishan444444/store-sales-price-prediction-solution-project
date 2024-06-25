from src.GoogleAnalyticsCustomerRevenuePrediction.components.data_ingestion import Dataingestion

import os
import sys
from src.GoogleAnalyticsCustomerRevenuePrediction.logger import logging
from src.GoogleAnalyticsCustomerRevenuePrediction.exception import customexception
import pandas as pd

obj=Dataingestion()

obj.initiate_data_ingestion()