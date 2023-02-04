"""
Authors: Sonia Castro Paniello
         Olga Villar Cairó
Data: 11/01/23
"""
"""
DATA ANALYSIS PIPELINE STEPS:
    (1) Add weights to balance the class frequencies.
    (2) Preprocessing
    (3) Perform cross validation to train the model avoiding overfitting.
    (4) Get the model predictions.
    (5) Compute model performance mesures to check its effectiveness.
"""

import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, DateType, StringType, DoubleType
from datetime import datetime
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#establishes spark session
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.8"
PYSPARK_DRIVER_PYTHON = "python3.8"

if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("BDAlab") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

labeled_data = spark.read.parquet('/mnt/c/users/sonia/desktop/CodeSkeleton/labeled_data.parquet')

#STEP (1)
#balances the classes to get better results by adding the following weight to each sample: # of rows / (# of rows of the class * 2)
zero_count_row = labeled_data.select(f.count(f.col("label") == 0).alias("zero_count")).collect()[0]
one_count_row= labeled_data.select(f.count(f.col("label") == 1).alias("one_count")).collect()[0]
zero_count = zero_count_row["zero_count"]
one_count = one_count_row["one_count"]
num_rows = labeled_data.count()
labeled_data = labeled_data.withColumn("weight", f.when(labeled_data.label == 0, num_rows/(zero_count*2)).otherwise(num_rows/(one_count*2)))

#STEP (2)
#unites the feature columns into one column adapting it to the model
feature_columns = ['Sensor_avg','flighthours', 'flightcycles', 'delayedminutes']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
labeled_data = assembler.transform(labeled_data)

#splits the dataset into training and testing
training_data, testing_data = labeled_data.randomSplit([0.8, 0.2])

#STEP (3)
#defines the cross validation parameters with the weights regularization
classifier = DecisionTreeClassifier(labelCol='label', featuresCol='features', weightCol='weight')
paramGrid = ParamGridBuilder() \
    .addGrid(classifier.maxDepth, [2, 3, 4]) \
    .addGrid(classifier.minInstancesPerNode, [14, 17, 20]) \
    .build()
evaluator = MulticlassClassificationEvaluator(labelCol='label')
cv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

#trains the model performing cross validation
model = cv.fit(training_data)

#STEP (4): computes all predictions (including the ones from training to see whether the model overfits)
predictions = model.transform(testing_data)
train_results = model.transform(training_data)

#STEP (5)
#computes the accuracy of the model on the training data
accuracy_evaluator_train = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
accuracy_train = accuracy_evaluator_train.evaluate(train_results)
print('Accuracy_train:', accuracy_train)

#computes the recall of the model on the training data
recall_evaluator_train = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderPR')
recall_train = recall_evaluator_train.evaluate(train_results)
print('Recall_train:', recall_train)

#computes the accuracy of the model on the testing data
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
accuracy = accuracy_evaluator.evaluate(predictions)
print('Accuracy:', accuracy)

#computes the recall of the model on the testing data
recall_evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderPR')
recall = recall_evaluator.evaluate(predictions)
print('Recall:', recall)

#saves the model to a file
model.write().overwrite().save("/mnt/c/users/sonia/desktop/CodeSkeleton/model.bin")
