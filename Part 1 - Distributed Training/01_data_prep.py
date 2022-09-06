# Databricks notebook source
# MAGIC %md
# MAGIC # 01. Data Preparation
# MAGIC 
# MAGIC In this notebook we will be working with the [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) from TensorFlow. It contains five classes of flower photos stored in JPEG format under five class-specific sub-directories. The images are hosted under [databricks-datasets](https://docs.databricks.com/data/databricks-datasets.html) at `dbfs:/databricks-datasets/flower_photos` for easy access.
# MAGIC 
# MAGIC We will:
# MAGIC - Load the JPEG files using Spark's binary file data source reader.
# MAGIC - Create a Delta table of the processed images in binary format.
# MAGIC - Extract labels from the image filepaths.
# MAGIC - Split our dataset into train and validation datasets for model training.
# MAGIC - Extract label indexes for each class and save train and validation datasets as Delta tables

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import io
import os
import shutil
import numpy as np
import pandas as pd
import mlflow

import pyspark.sql.functions as f

# COMMAND ----------

# MAGIC %md
# MAGIC ## i. Convert raw JPEG files to Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a quick look at the files in DBFS

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/flower_photos/daisy

# COMMAND ----------

# MAGIC %md
# MAGIC We will use Spark's [binary file](https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/binary-file) data source reader to load the JPEG files from [DBFS](https://docs.databricks.com/data/databricks-file-system.html) into a Spark DataFrame.
# MAGIC 
# MAGIC This will read take the binary files and convert each file into a single record that contains the raw content and metadata of the file. The binary file data source produces a DataFrame with the following columns and possibly partition columns:
# MAGIC 
# MAGIC * `path` (StringType): The path of the file.
# MAGIC * `modificationTime` (TimestampType): The modification time of the file. In some Hadoop FileSystem implementations, this parameter might be unavailable and the value would be set to a default value.
# MAGIC * `length` (LongType): The length of the file in bytes.
# MAGIC * `content` (BinaryType): The contents of the file.
# MAGIC 
# MAGIC Using the `pathGlobFilter` option we will load files with paths matching a given glob pattern while keeping the behavior of partition discovery. The following code reads all JPG files from the input directory with partition discovery.
# MAGIC 
# MAGIC We will also use the `recursiveFileLookup` option to ignore partition discovery and recursively search files under the input directory. This option searches through nested directories even if their names do not follow a partition naming scheme like `date=2019-07-01`. The following cell reads all JPEG files recursively from the input directory and ignores partition discovery.

# COMMAND ----------

bronze_df = (spark.read.format('binaryFile')
             .option('pathGlobFilter', '*.jpg')
             .option('recursiveFileLookup', 'true')
             .load('/databricks-datasets/flower_photos/')
             .sample(fraction=0.5)    # Sample to speed up training
            )

bronze_df.display()

# COMMAND ----------

bronze_df.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ii. Creating Bronze Delta Table from Spark Dataframe

# COMMAND ----------

bronze_tbl_name = 'bronze'

# Delete the old database and tables if needed (for demo purposes)
spark.sql(f'DROP DATABASE IF EXISTS {database_name} CASCADE')

# Create database to house tables
spark.sql(f'CREATE DATABASE {database_name}')

# COMMAND ----------

# To improve read performance when you load data back, Databricks recommends turning off compression when you save data loaded from binary files
spark.conf.set('spark.sql.parquet.compression.codec', 'uncompressed')

# Create a Delta Lake table from loaded 
bronze_df.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{bronze_tbl_name}')

# COMMAND ----------

spark.sql(f'SELECT * FROM {database_name}.{bronze_tbl_name}').display()

# COMMAND ----------

# Load bronze table for faster transformations
bronze_df = spark.table(f'{database_name}.{bronze_tbl_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC Reading from a Delta table is much faster than trying to load all the images from a directory. By saving the data in the Delta format, we can continue to process the images into a silver table, transforming it into a proper training set for our use case. This entails parsing out labels from the image paths and converting string labels into numerical values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## iii. Create Silver Table from Bronze Table

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will use a [pandas UDF](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) to extract the labels for each image from the filepath. 

# COMMAND ----------

# Define pandas UDF to extract the image label from the file path
@f.pandas_udf('string')
def get_label_udf(path_col: pd.Series) -> pd.Series:
    return path_col.map(lambda path: path.split('/')[-2])

# Apply pandas UDF to path column
silver_df = bronze_df.withColumn('label', get_label_udf('path'))

# COMMAND ----------

# Save table with label as silver table
silver_tbl_name = 'silver'
silver_df.write.format('delta').mode('overwrite').saveAsTable(f'{database_name}.{silver_tbl_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iv. Split dataset into train and validation sets

# COMMAND ----------

# MAGIC %md
# MAGIC Prior to model training, we want to split our datasets into a `train` and `val` dataset.

# COMMAND ----------

# Load dataset from the Silver table
dataset_df = spark.table(f'{database_name}.{silver_tbl_name}')
display(dataset_df)

# COMMAND ----------

# MAGIC %md
# MAGIC For testing purposes, we'll use a subset of the data and split it into a `train` and `val` dataset.

# COMMAND ----------

train_df, val_df = dataset_df.randomSplit([0.9, 0.1], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## v. Add label index column to train and validation datasets

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will need to pass our label as a numeric value during training. As such, we create a label index prior to writing out our train and test datasets.

# COMMAND ----------

# Create index lookups for labels
labels = train_df.select(f.col('label')).distinct().collect()
label_to_idx = {label: index for index, (label, ) in enumerate(sorted(labels))}

print(label_to_idx)

# COMMAND ----------

# Define UDF to extract the image index
@f.pandas_udf('int')
def get_label_idx_udf(labels_col: pd.Series) -> pd.Series:
    return labels_col.map(lambda label: label_to_idx[label])

# Create train DataFrame with label index column
train_df = (train_df
            .withColumn('label_idx', get_label_idx_udf('label')))

# Create val DataFrame with label index column
val_df = (val_df
          .withColumn('label_idx', get_label_idx_udf('label')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## vi. Save train and val datasets 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's create tables from our train and validation datasets.

# COMMAND ----------

silver_train_tbl_name = 'silver_train'
silver_val_tbl_name = 'silver_val'

(train_df.write.format('delta')
 .mode('overwrite')
 .saveAsTable(f'{database_name}.{silver_train_tbl_name}'))

(val_df.write.format('delta')
 .mode('overwrite')
 .saveAsTable(f'{database_name}.{silver_val_tbl_name}'))

# COMMAND ----------

# MAGIC %md
# MAGIC Nice! We now have our data ready for model training. In the [next notebook]($./02_model_training_single_node) we will look at training a model in a single node setting, using the Delta tables we just created.
