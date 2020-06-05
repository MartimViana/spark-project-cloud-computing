

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer


spark=SparkSession.builder.appName("lrexample").getOrCreate()
data=spark.read.csv("price_paid_records-000.csv", inferSchema=True, header=True)

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
data.show
assembler=VectorAssembler(inputCols=['Price', 'Date_of_TransferIndexed', 'Property_typeIndexed', 'Old_NewIndexed', 'TownIndexed', 'DistrictIndexed', 'CountyIndexed'],outputCol='features')
output=assembler.transform(data)
final_data=output.select('features','Price')
train_data,test_data=final_data.randomSplit([0.7,0.3])

lr=LinearRegression(labelCol='Price')
lr_model=lr.fit(train_data)

# save results
filename = 'Machine_Learning'
lr_model.save(filename)
