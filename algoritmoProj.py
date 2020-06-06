from pyspark.sql import SparkSession
from pyspark.sql.functions import year, mean
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
#from pyspark.ml.classification import RandomForestClassifier
#from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import classification_report, confusion_matrix 

import urllib, json

import pandas

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression

import pandas
import joblib


def drawGraphic(data, kind1, colX, colY, title, xlabel, ylabel):
    
    axTemp = plt.gca()
    for d in data:
        df = d.toPandas()
        df.plot(kind=kind1, x=colX, y=colY, ax=axTemp)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.autoscale(tight=True)
    return plt
    
def drawLineGraphic(data, colX, colY, title, xlabel, ylabel):
    return drawGraphic(data, 'bar', colX, colY, title, xlabel, ylabel)

spark = SparkSession.builder.getOrCreate()

data = spark.read.csv('price_paid_records-000.csv', header=True, inferSchema=True).persist()
data.show() 
features = ['Price', 'Date of Transfer', 'Property Type', 'Old/New', 'Town/City', 'District', 'County']
data = data.select(features)
# convert all selected string columns into integers
date_indexer = StringIndexer(inputCol='Date of Transfer', outputCol='Date_of_TransferIndexed')
date_indexer = date_indexer.fit(data)
property_type_indexer = StringIndexer(inputCol='Property Type', outputCol='Property_typeIndexed')
property_type_indexer = property_type_indexer.fit(data)
olde_new_indexer = StringIndexer(inputCol='Old/New', outputCol='Old_NewIndexed')
olde_new_indexer = olde_new_indexer.fit(data)
town_indexer = StringIndexer(inputCol='Town/City', outputCol='TownIndexed')
town_indexer = town_indexer.fit(data)
district_indexer = StringIndexer(inputCol='District', outputCol='DistrictIndexed')
district_indexer = district_indexer.fit(data)
county_indexer = StringIndexer(inputCol='County', outputCol='CountyIndexed')
county_indexer = county_indexer.fit(data)
data = date_indexer.transform(data)
data = property_type_indexer.transform(data)
data = olde_new_indexer.transform(data)
data = town_indexer.transform(data)
data = district_indexer.transform(data)
data = county_indexer.transform(data)
    

# process data
                                        # convert datetime to integer
assembler = VectorAssembler().setInputCols(['DistrictIndexed', 'Property_typeIndexed', 'Date_of_TransferIndexed' ]).setOutputCol('features')    # set input and output columns
data = assembler.transform(data)
data=data.select('features','Price')
    

   # remove unecessary columns
data = data[['features', 'Price']]
#data = data[['DistrictIndexed', 'Property_typeIndexed', 'Date_of_TransferIndexed', 'Price']]
data.show()
    
    # split data
seed = 1234
train, test = data.randomSplit([0.6, 0.4], seed)


    # perform linear regression
algorithm = LinearRegression( labelCol='Price')
model = algorithm.fit(train)
result = model.transform(test)

    # show results
result.show()

 # save results
filename = 'modelofds'
model.save(filename)
#result.repartition(1).write.format('com.databricks.spark.csv').save('C:/Users/andre/OneDrive/Desktop/result.csv',header = 'true')
#avg_price_county = (data
#	    .groupBy("DistrictIndexed")
#	    .agg(mean("Price").alias("Price_mean")))

#drawLineGraphic([avg_price_county],'DistrictIndexed', 'Price_mean', 'title', 'Distric','Price' )
#plt.savefig('C:/Users/andre/OneDrive/Desktop/price_per_regien.png')




#evaluation_summary = model.evaluate(test)
#print("Mean absolute error: "+str(evaluation_summary.meanAbsoluteError))
#print("Root mean squared error: "+str(evaluation_summary.rootMeanSquaredError))
#print("R2: "+str(evaluation_summary.r2))