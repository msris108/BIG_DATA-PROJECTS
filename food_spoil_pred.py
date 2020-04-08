from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier

'''
You've been hired by a dog food company to try to predict why some batches of their dog food are spoiling much quicker than intended!
Unfortunately this Dog Food company hasn't upgraded to the latest machinery, 
meaning that the amounts of the five preservative chemicals they are using can vary a lot, 
but which is the chemical that has the strongest effect? 
The dog food company first mixes up a batch of preservative that contains 4 different preservative chemicals (A,B,C,D) 
and then is completed with a "filler" chemical. The food scientists beelive one of the A,B,C, or D preservatives is causing the problem, 
but need your help to figure out which one! Use Machine Learning with RF to find out which parameter had the most predicitive power, 
thus finding out which chemical causes the early spoiling! So create a model and then find out how you can decide which chemical is the problem!

    Pres_A : Percentage of preservative A in the mix
    Pres_B : Percentage of preservative B in the mix
    Pres_C : Percentage of preservative C in the mix
    Pres_D : Percentage of preservative D in the mix
    Spoiled: Label indicating whether or not the dog food batch was spoiled.

'''

spark = SparkSession.builder.appName('dogfood').getOrCreate()

data = spark.read.csv('dog_food.csv',inferSchema=True,header=True)

data.printSchema()
data.head()
data.describe().show()

assembler = VectorAssembler(inputCols=['A', 'B', 'C', 'D'],outputCol="features")
output = assembler.transform(data)

rfc = DecisionTreeClassifier(labelCol='Spoiled',featuresCol='features')
output.printSchema()

final_data = output.select('features','Spoiled')
final_data.head()

rfc_model = rfc.fit(final_data)

print(rfc_model.featureImportances)


# Feature at index 2 (Chemical C) is by far the most important feature, meaning it is causing the early spoilage!