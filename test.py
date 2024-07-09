from src.GoogleAnalyticsCustomerRevenuePrediction.pipelines.prediction_pipeline import CustomData
from src.GoogleAnalyticsCustomerRevenuePrediction.pipelines.prediction_pipeline import PredictPipeline


obj=CustomData(15,	6.865,	'Regular'	,0.035186,	'Dairy',	249.8092,	'Medium','Tier 2',	'Supermarket Type2'	)

final_data=obj.get_data_as_dataframe()
        
predict_pipeline=PredictPipeline()
        
pred=predict_pipeline.Predict(final_data)
        
result=pred

print(final_data)
print(result)