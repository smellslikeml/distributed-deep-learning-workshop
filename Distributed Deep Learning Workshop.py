# Databricks notebook source
# MAGIC %md
# MAGIC *Inspired by Sean Owen's [Blog](https://databricks.com/blog/2019/08/15/how-not-to-scale-deep-learning-in-6-easy-steps.html) and [Notebook](https://demo.cloud.databricks.com/#notebook/10675022), How (Not) to Scale Deep Learning in 6 Easy steps*

# COMMAND ----------

# MAGIC %md
# MAGIC # Distributed Deep Learning on Databricks
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Deep Learning continues to be an exciting and powerful tool for a large range of problems. It requires a large amount of compute, and often, users find that deep learning model training seems 'too slow'. Sometimes this is due to common, simple errors that can be avoided. But scaling deep learning also requires a different understanding of how to trade off speed, efficiency, and accuracy, and a new set of tools like Tensorflow, Petastorm, and Horovod.
# MAGIC 
# MAGIC This notebooks walks through training an image classifier on the flowers dataset starting with single node training and progressing to distributed training with GPU clusters.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC 
# MAGIC This loads the [flowers dataset](https://databricks.com/notebooks/simple-aws/petastorm-spark-converter-tensorflow.html#:~:text=We%20use%20the-,flowers%20dataset,-from%20the%20TensorFlow) hosted by the Tensorflow team. It contains flower photos stored under five sub-directories, one per class. It is hosted under databricks-datasets `dbfs:/databricks-datasets/flower_photos` for easy access.
# MAGIC 
# MAGIC In this example, we load the flowers table which contains the preprocessed flowers dataset using binary file data source. We will use a small subset of the flowers dataset, including ~900 train images, and ~100 validation images.

# COMMAND ----------

import io, os
import numpy as np
import mlflow

from pyspark.sql.functions import col

# COMMAND ----------

# let's take a quick look at one of our image file directory
dbutils.fs.ls("/databricks-datasets/flower_photos/daisy")

# COMMAND ----------

bronze_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg").option("recursiveFileLookup", "true").load("/databricks-datasets/flower_photos/")
display(bronze_df)

# COMMAND ----------

# Load libraries
import shutil

# Set config for database name, file paths, and table names
database_name = 'distributed_dl_workshop'

# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# Paths for various Delta tables
bronze_tbl_path = '/home/{}/distributed_dl_workshop/bronze/'.format(user)
silver_tbl_path = '/home/{}/distributed_dl_workshop/silver/'.format(user)

bronze_tbl_name = 'bronze'
silver_tbl_name = 'silver'

# Delete the old database and tables if needed (for demo purposes)
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating Bronze Delta Table from Spark Dataframe

# COMMAND ----------

#To improve read performance when you load data back, Databricks recommends turning off compression when you save data loaded from binary files:
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

# Create a Delta Lake table from loaded 
bronze_df.write.format('delta').mode('overwrite').save(bronze_tbl_path)

# COMMAND ----------

# Create bronze table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,bronze_tbl_name,bronze_tbl_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM distributed_dl_workshop.bronze

# COMMAND ----------

# load as a bronze table for faster transformations
bronze_df = spark.table("distributed_dl_workshop.bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC Reading from a Delta table is much faster than trying to load all the images from a directory! By saving the data in the Delta format, we can continue to process the images into a silver table, transforming it into a proper training set for our use case. This entails parsing out labels from the image paths and converting string labels into numerical values.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Silver Table from Bronze Table

# COMMAND ----------

#custom udf to extract labels out of image paths
from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer

# convert string categories to numerical 
indexer = StringIndexer(inputCol="category", outputCol="label")


@udf("string")
def extract_category(path):
  return path.split("/")[-2]

df = bronze_df.withColumn("category", extract_category("path"))
silver_df = indexer.fit(df).transform(df)
display(silver_df)

# COMMAND ----------

# save as delta table
silver_df.write.format('delta').mode('overwrite').save(silver_tbl_path)

# Create silver table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_tbl_name,silver_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Having transformed and saved our data into a silver table, we can select the columns we'll need for training.

# COMMAND ----------

dataset_df = spark.table("distributed_dl_workshop.silver").select("content", "label")
display(dataset_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's get the number of possible labels we'll need to classify. For testing purposes, we'll use a subset of the data and split it into a `train` and `val` dataset.

# COMMAND ----------

img_size = 224
num_classes = dataset_df.select("label").distinct().count()
df_train, df_val = dataset_df.randomSplit([0.9, 0.1], seed=12345)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save our `train` and `val` datasets into delta lake tables for future use.

# COMMAND ----------

#split our silver table into proper train and val tables

silver_train_tbl_path = '/home/{}/distributed_dl_workshop/silver_train/'.format(user)
silver_val_tbl_path = '/home/{}/distributed_dl_workshop/silver_val/'.format(user)

silver_train_tbl_name = 'silver_train'
silver_val_tbl_name = 'silver_val'


# Drop any old delta lake files if needed (e.g. re-running this notebook with the same paths)
shutil.rmtree('/dbfs'+silver_train_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_val_tbl_path, ignore_errors=True)

# save as delta table
df_train.write.format('delta').mode('overwrite').save(silver_train_tbl_path)
df_val.write.format('delta').mode('overwrite').save(silver_val_tbl_path)

# Create silver table
_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_train_tbl_name,silver_train_tbl_path))

_ = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_val_tbl_name,silver_val_tbl_path))

# COMMAND ----------

# MAGIC %md
# MAGIC To train a model with Tensorflow, we'll need to transform our spark dataframes into `tf.data.Dataset`s. Tensorflow has a function to transform a pandas dataframe into a [tensorflow dataset](https://www.tensorflow.org/datasets).
# MAGIC For this example, we'll convert our dataframe to a pandas dataframe using `.toPandas()` and convert to a properly formated tf dataset from there.

# COMMAND ----------

import tensorflow as tf

features = ["content" "label"]

pd_train = df_train.toPandas() 
pd_val = df_val.toPandas()

train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (pd_train["content"].values,pd_train["label"].values))
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (pd_val["content"].values,pd_val["label"].values))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can continue transforming our image dataset into the correct format using a preprocessing function and mapping it to our `tf.datasets`

# COMMAND ----------

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

batch_size = 32

def preprocess(content, label):
  """
  Preprocess an image file bytes for MobileNetV2 (ImageNet).
  """
  image = tf.image.decode_jpeg(content, channels=3)
  image = tf.image.resize(image, [img_size,img_size])
  return preprocess_input(image), label

train_dataset = train_dataset.map(lambda content, label: preprocess(content, label)).batch(batch_size)
val_dataset = val_dataset.map(lambda content, label: preprocess(content, label)).batch(batch_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single Node Training
# MAGIC 
# MAGIC Single node training is the most common way machine learning practitioners set up their training regimes. For some modeling use cases, this is a great way to go as all the modeling can stay on one machine with no additional libraries. 
# MAGIC 
# MAGIC Training with Databricks is just as easy. Simply use a [Single Node Cluster](https://docs.databricks.com/clusters/single-node.html) which is just a Spark Driver with no worker nodes. 

# COMMAND ----------

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Additional options here will be used later:

def build_model():
  base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3))
  
  #freeze base model layers
  for layer in base_model.layers:
    layer.trainable = False
    
  model = Sequential([
    base_model, 
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
  ])

  return model

# COMMAND ----------

# mlflow autologging
mlflow.tensorflow.autolog()

model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 3
steps_per_epoch = len(train_dataset) // batch_size
validation_steps = len(val_dataset) // batch_size

model.fit(train_dataset,
                   steps_per_epoch = steps_per_epoch,
                   epochs = epochs,
                   verbose = 1,
                   validation_data = val_dataset,
                   validation_steps = validation_steps)

# COMMAND ----------

# MAGIC %md
# MAGIC Wohoo! We trained a model on a single machine very similarly to what you might have done on your laptop. It was easy for us to convert this small dataset into a pandas dataframe from a delta table and ultimately convert it into a tensorflow dataset for training. 
# MAGIC 
# MAGIC However, most production workloads will require training with **orders of magnitude** more data, so much so that it could overwhelm a single machine while training. Other cases could contribute to exhausting a single node during training such as training large model architectures.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Training
# MAGIC 
# MAGIC A very large training problem may benefit from scaling out verticaly (training on a larger instance) or horizontally (using multiple machines). Scaling vertically may be an approach to adopt, but it can be costly to run a large instance for an extended period of time and scaling a single machine is limited (ie. there's only so many cores + RAM one machine could have). 
# MAGIC 
# MAGIC Scaling horizontally can be an affordable way to leverage the compute required to tackle very large training events. Multiple smaller instances may be more readibily available at cheaper rates in the cloud than a single, very large instance. Theoretically, you could add as many nodes as you wish (dependent on your cloud limits). In this section, we'll go over how to incorporate [Petastorm](https://docs.databricks.com/applications/machine-learning/load-data/ddl-data.html) and [Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) into our single node training regime to distribute training to multiple machines.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Loading data with Petastorm
# MAGIC 
# MAGIC Our first single node training example only used a fraction of the data and it required to work with the data sitting in memory. Typical datasets used for training may not fit in memory for a single machine. Petastorm enables directly loading data stored in parquet format, meaning we can go from our silver Delta table to a distributed `tf.data.Dataset` without having to copy our table into a `Pandas` dataframe and wasting additional memory.
# MAGIC 
# MAGIC First, let's split our silver table into train and val tables:

# COMMAND ----------

# MAGIC %md
# MAGIC Here we use `petastorm` to load and cache data that was read directly from our silver train and val Delta tables:

# COMMAND ----------

df_train = spark.read.format("delta").load(silver_train_tbl_path)
df_val = spark.read.format("delta").load(silver_val_tbl_path)

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
df_train = df_train.repartition(2)
df_val = df_val.repartition(2)

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

dbutils.fs.rm("/tmp/distributed_dl_workshop/petastorm", recurse=True)
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/distributed_dl_workshop/petastorm")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

train_size = len(converter_train)
val_size = len(converter_val)

# COMMAND ----------

with converter_train.make_tf_dataset(batch_size=batch_size) as train_ds,\
     converter_val.make_tf_dataset(batch_size=batch_size) as val_ds:
  
  # mlflow autologging
  mlflow.tensorflow.autolog()
  
  # Transforming datasets to map our preprocess function() and then batch()
  train_ds = train_ds.unbatch().map(lambda x: (x.content, x.label))
  val_ds = val_ds.unbatch().map(lambda x: (x.content, x.label))
  
  train_ds = train_ds.map(lambda content, label: preprocess(content, label)).batch(batch_size)
  val_ds = val_ds.map(lambda content, label: preprocess(content, label)).batch(batch_size)
  
  model = build_model()
  model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  epochs = 3
  steps_per_epoch = train_size // batch_size
  validation_steps = val_size // batch_size

  model.fit(train_ds,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            verbose = 1,
            validation_data = val_ds,
            validation_steps = validation_steps)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see the code to train our model is exactly the same as our single node training. Adding Petastorm to load our data directly from the underlying parquet files in our delta tables simply required using the `make_spark_converter` class and initializing our datasets using the `make_tf_dataset()` method.
# MAGIC 
# MAGIC Next, we need to distribute the training across our cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Distribute training with Horovod
# MAGIC 
# MAGIC [Horovod](https://github.com/horovod/horovod) is a distributed training framework for TensorFlow, Keras, and PyTorch. Databricks supports distributed deep learning training using HorovodRunner and the `horovod.spark` package which we will use next. 
# MAGIC 
# MAGIC We can use Horovod to train across multiple machines, meaning we can distribute training across CPU-clusters or GPU clusters. 

# COMMAND ----------

# MAGIC %md
# MAGIC [HorovodRunner](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html#:~:text=HorovodRunner%20is%20a%20general%20API,learning%20training%20jobs%20on%20Spark.) takes a Python method that contains deep learning training code with Horovod hooks. HorovodRunner pickles the method on the driver and distributes it to Spark workers. Essentially, this API lets you launch Horovod training jobs as Spark jobs! To be able to work with HorovodRunner correcty, we need to package all of our logic into a function. This will ensure the logic is distributed across each node.

# COMMAND ----------

# MAGIC %md
# MAGIC Inside our function, we'll incorporate the same logic we used in the single node and petastorm training examples plus additional logic to account for GPUs. 
# MAGIC 
# MAGIC If you have a CPU cluster, then these lines will be ignored and training will occur on CPU.

# COMMAND ----------

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

databricks_host = 'https://field-eng.cloud.databricks.com/'
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def train_and_evaluate_hvd():
  hvd.init()  # Initialize Horovod.
  
  mlflow.mlflow.set_tracking_uri('databricks')
  os.environ['DATABRICKS_HOST'] = databricks_host
  os.environ['DATABRICKS_TOKEN'] = databricks_token
  
  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
      
  # Creating model
  model = build_model()
  
  # Horovod: adjust learning rate based on number of workers.
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
  dist_optimizer = hvd.DistributedOptimizer(optimizer)
  
  callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
  ]
  
  # Set experimental_run_tf_function=False in TF 2.x
  model.compile(optimizer=dist_optimizer, 
                loss="sparse_categorical_crossentropy", 
                metrics=["accuracy"],
                experimental_run_tf_function=False)
  
    
  # same logic as Petastorm training example here with slight adjustments
  with converter_train.make_tf_dataset(batch_size=batch_size, cur_shard=hvd.rank(), shard_count=hvd.size()) as train_ds,\
     converter_val.make_tf_dataset(batch_size=batch_size, cur_shard=hvd.rank(), shard_count=hvd.size()) as val_ds:
    
    # Transforming datasets to map our preprocess function() and then batch()
    train_ds = train_ds.unbatch().map(lambda x: (x.content, x.label))
    val_ds = val_ds.unbatch().map(lambda x: (x.content, x.label))
  
    train_ds = train_ds.map(lambda content, label: preprocess(content, label)).batch(batch_size)
    val_ds = val_ds.map(lambda content, label: preprocess(content, label)).batch(batch_size)

    epochs = 3
    steps_per_epoch = train_size // (batch_size * hvd.size())
    validation_steps = max(1, val_size // (batch_size * hvd.size()))

    hist = model.fit(train_ds,
              steps_per_epoch = steps_per_epoch,
              epochs = epochs,
              verbose = 1,
              validation_data = val_ds,
              validation_steps = validation_steps)
    # MLflow Tracking (Log only from Worker 0)
    if hvd.rank() == 0:
      # Log events to MLflow
      with mlflow.start_run(run_id = active_run_uuid):
        # Log MLflow Parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        # Log MLflow Metrics
        mlflow.log_metric("val Loss", hist.history['val_loss'][-1])
        mlflow.log_metric("val Accuracy", hist.history['val_accuracy'][-1])
      
        # Log Model
        mlflow.keras.log_model(model, "models")
  
  return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]

# COMMAND ----------

with mlflow.start_run() as run:
  active_run_uuid = mlflow.active_run().info.run_uuid
  hr = HorovodRunner(np=2, driver_log_verbosity='all')  # This assumes the cluster consists of two workers.
  hr.run(train_and_evaluate_hvd)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring and Optimizing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single node vs Distributed Training Cosiderations
# MAGIC 
# MAGIC Whether you choose to train in a distributed manner or using a single VM will be highly dependent on the scale of your training data and architecture complexity + size. There may be instances where leveraging Petastorm will help get past memory issues around dataset loading but distributing training across a cluster may add too much additional training time and/or underutilized resources.
# MAGIC 
# MAGIC Petastorm is very useful in situations were training data is the bottleneck. Distributing the dataset across the cluster is a great way to get past the memory limitations of a single node.
# MAGIC 
# MAGIC Horovod allows us to leverage the compute of multiple machines OR multiple devices (like GPUs) during training. This trait is especially useful when training with large architectures and/or massive datasets. At times leveraging multiple machines with a single GPU could be more cost effective than using a single, very large machine with multiple GPUs. 

# COMMAND ----------

# MAGIC %md
# MAGIC Given all of these options, how do you best tune the resources you'll need for your training needs?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ganglia Metrics
# MAGIC 
# MAGIC Found under the **Metrics** tab, [Ganglia](https://docs.databricks.com/clusters/clusters-manage.html#monitor-performance) live metrics and historical snapshots provide a view into how cluster resources are utilized at any point in time. 
# MAGIC 
# MAGIC ![ganglia metrics](https://docs.databricks.com/_images/metrics-tab.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimizations
# MAGIC 
# MAGIC Understanding how to interpret Ganglia metrics, you can observe these as you experiment with techniques that can further optimize your training.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Early Stopping
# MAGIC 
# MAGIC Early stopping can prevent you from potentially overfitting and/or waste compute resources by monitoring a metric of interest. In this example we may measure `val_accuracy`, to end training early when the metric no longer shows improvement. You can dial the sensitivity of this setting by adjusting the `patience` parameter.
# MAGIC 
# MAGIC Add this as part of your `callbacks` parameter in the `fit()` method.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```python
# MAGIC 
# MAGIC # add Early Stopping as a list for the callbacks parameter
# MAGIC callbacks = [
# MAGIC   tf.keras.callbacks.EarlyStopping(
# MAGIC     monitor="val_loss",
# MAGIC     patience=3 # Number of epochs with no improvement after which training will be stopped
# MAGIC   )
# MAGIC ]
# MAGIC 
# MAGIC # ...
# MAGIC 
# MAGIC hist = model.fit(train_ds,
# MAGIC               steps_per_epoch = steps_per_epoch,
# MAGIC               epochs = epochs,
# MAGIC               verbose = 1,
# MAGIC               validation_data = val_ds,
# MAGIC               validation_steps = validation_steps,
# MAGIC               callbacks=callbacks)
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Larger Batch sizes
# MAGIC 
# MAGIC You can improve training speed and potentially [boost model performance](https://arxiv.org/abs/2012.08795) by increasing the batch size during training. Larger batch sizes can help stabilize training but may take more resources (memory + compute) to load and process. If you find you are underwhelming your cluster resources, this could be a good knob to turn to increase utilization and speed up training. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adapt Learning Rate
# MAGIC 
# MAGIC A larger learning rate may also help the model converge to a local minima faster by allowing the model weights to change more dramatically between epochs. Although it *may* find a local minima faster, this could lead to a suboptimal solution or even derail training all together where the model may not be able to converge. 
# MAGIC 
# MAGIC This parameter is a powerful one to experiment with and machine learning practitioners should be encouraged to do so! One can tune this by trial and error or by implementing a scheduler that can vary the parameter depending on duration of training or in response to some observed training progress.
# MAGIC 
# MAGIC More on the learning rate parameter can be found in this [reference](https://amzn.to/2NJW3gE).
