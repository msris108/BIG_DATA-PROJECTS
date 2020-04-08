from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

'''
DATA:
    'Session_Connection_Time': How long the session lasted in minutes
    'Bytes Transferred': Number of MB transferred during session
    'Kali_Trace_Used': Indicates if the hacker was using Kali Linux
    'Servers_Corrupted': Number of server corrupted during the attack
    'Pages_Corrupted': Number of pages illegally accessed
    'Location': Location attack came from (Probably useless because the hackers used VPNs)
    'WPM_Typing_Speed': Their estimated typing speed based on session logs.

The technology firm has 3 potential hackers that perpetrated the attack.
Their certain of the first two hackers but they aren't very sure if the third hacker was involved or not.
They have requested your help! Can you help figure out whether or not the third suspect had anything to do with the attacks,
or was it just two hackers? It's probably not possible to know for sure, but maybe what you've just learned about Clustering can help!

One last key fact, the forensic engineer knows that the hackers trade off attacks.
Meaning they should each have roughly the same amount of attacks.
For example if there were 100 total attacks, then in a 2 hacker situation each should have about 50 hacks, 
in a three hacker situation each would have about 33 hacks.
The engineer believes this is the key element to solving this,
but doesn't know how to distinguish this unlabeled data into groups of hackers.
'''

spark = SparkSession.builder.appName('hack_find').getOrCreate()
dataset = spark.read.csv("hack_data.csv",header=True,inferSchema=True)

dataset.head()
dataset.describe().show()
df.columns

feat_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used',
             'Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed']

vec_assembler = VectorAssembler(inputCols = feat_cols, outputCol='features')
final_data = vec_assembler.transform(dataset)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scalerModel = scaler.fit(final_data)

cluster_final_data = scalerModel.transform(final_data)

kmeans3 = KMeans(featuresCol='scaledFeatures',k=3)
kmeans2 = KMeans(featuresCol='scaledFeatures',k=2)

model_k3 = kmeans3.fit(cluster_final_data)
model_k2 = kmeans2.fit(cluster_final_data)

wssse_k3 = model_k3.computeCost(cluster_final_data)
wssse_k2 = model_k2.computeCost(cluster_final_data)

print("With K=3")
print("Within Set Sum of Squared Errors = " + str(wssse_k3))
print('--'*30)
print("With K=2")
print("Within Set Sum of Squared Errors = " + str(wssse_k2))

for k in range(2,9):
    kmeans = KMeans(featuresCol='scaledFeatures',k=k)
    model = kmeans.fit(cluster_final_data)
    wssse = model.computeCost(cluster_final_data)
    print("With K={}".format(k))
    print("Within Set Sum of Squared Errors = " + str(wssse))
    print('--'*30)

model_k3.transform(cluster_final_data).groupBy('prediction').count().show()

model_k2.transform(cluster_final_data).groupBy('prediction').count().show()