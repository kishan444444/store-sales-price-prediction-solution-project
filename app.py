from src.GoogleAnalyticsCustomerRevenuePrediction.pipelines.prediction_pipeline import PredictPipeline
from src.GoogleAnalyticsCustomerRevenuePrediction.pipelines.prediction_pipeline import CustomData


from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        obj=CustomData(
                        Item_Identifier=int(request.form.get("Item Identifier")),
                        Item_Weight=float(request.form.get("Item Weight")),
                        Item_Fat_Content=request.form.get("Item Fat Content"),
                        Item_Visibility=float(request.form.get("Item Visibility")),
                        Item_Type=request.form.get("Item Type"),
                        Item_MRP=float(request.form.get("Item MRP")),
                        Outlet_Size=request.form.get("Outlet Size"),
                        Outlet_Location_Type=request.form.get("Outlet Location Type"),
                        Outlet_Type=request.form.get("Outlet Type")
        
        )
        
         # this is my final data
        final_data=obj.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.Predict(final_data)
        result=pred
        
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
