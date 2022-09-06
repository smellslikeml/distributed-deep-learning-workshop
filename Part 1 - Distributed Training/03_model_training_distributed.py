# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Model Training - Distributed 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook we will see how we can take our single node training code and scale training across our cluster.
# MAGIC 
# MAGIC We will:
# MAGIC - Use [Petastorm](https://docs.databricks.com/applications/machine-learning/load-data/ddl-data.html) to load our datasets from Delta and convert to tf.data datsets.
# MAGIC - Conduct single node training using Petastorm.
# MAGIC - Scale training across multiple nodes using [Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) to orchestrate training, with Petastorm as a means to feed data to the training process.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Motivation

# COMMAND ----------

# MAGIC %md
# MAGIC A very large training problem may benefit from scaling out verticaly (training on a larger instance) or horizontally (using multiple machines). Scaling vertically may be an approach to adopt, but it can be costly to run a large instance for an extended period of time and scaling a single machine is limited (ie. there's only so many cores + RAM one machine could have). 
# MAGIC 
# MAGIC Scaling horizontally can be an affordable way to leverage the compute required to tackle very large training events. Multiple smaller instances may be more readibily available at cheaper rates in the cloud than a single, very large instance. Theoretically, you could add as many nodes as you wish (dependent on your cloud limits). In this section, we'll go over how to incorporate [Petastorm](https://docs.databricks.com/applications/machine-learning/load-data/ddl-data.html) and [Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) into our single node training regime from to distribute training to multiple machines.

# COMMAND ----------

# MAGIC %md
# MAGIC Our first single node training example only used a fraction of the data and it required to work with the data sitting in memory. Typical datasets used for training may not fit in memory for a single machine. Petastorm enables directly loading data stored in parquet format, meaning we can go from our silver Delta table to a distributed `tf.data.Dataset` without having to copy our table into a `Pandas` dataframe and wasting additional memory.
# MAGIC 
# MAGIC <a href="https://github.com/uber/petastorm" target="_blank">Petastorm</a> enables single machine or distributed training and evaluation of deep learning models from datasets in Apache Parquet format and datasets that are already loaded as Spark DataFrames. It supports ML frameworks such as TensorFlow, Pytorch, and PySpark and can be used from pure Python code.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we use `petastorm` to load and cache data that was read directly from our silver train and val Delta tables:

# COMMAND ----------

# MAGIC %run ./00_setup 

# COMMAND ----------

import os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import mlflow

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## i. Configs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Set global variables

# COMMAND ----------

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

BATCH_SIZE = 256
EPOCHS = 3

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ii. Load prepared data from Silver layer 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load the train and validation datasets from the Silver layer we created in [01_data_prep]($./01_data_prep) as Spark DataFrames.

# COMMAND ----------

cols_to_keep = ['content', 'label_idx']

train_tbl_name = 'silver_train'
val_tbl_name = 'silver_val'

train_df = (spark.table(f'{database_name}.{train_tbl_name}')
            .select(cols_to_keep))

val_df = (spark.table(f'{database_name}.{val_tbl_name}')
          .select(cols_to_keep))

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
train_df = train_df.repartition(2)
val_df = val_df.repartition(2)

print('train_df count:', train_df.count())
print('val_df count:', val_df.count())

# COMMAND ----------

num_classes = train_df.select('label_idx').distinct().count()

print('Number of classes:', num_classes)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## iii. Convert the Spark DataFrame to a TensorFlow Dataset
# MAGIC In order to convert Spark DataFrames to a TensorFlow datasets, we need to do it in two steps:
# MAGIC <br><br>
# MAGIC 
# MAGIC 
# MAGIC 0. Define where you want to copy the data by setting Spark config
# MAGIC 0. Call `make_spark_converter()` method to make the conversion
# MAGIC 
# MAGIC This will copy the data to the specified path.

# COMMAND ----------

dbutils.fs.rm(f'/tmp/distributed_dl_workshop_{user}/petastorm', recurse=True)
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, f'file:///dbfs/tmp/distributed_dl_workshop_{user}/petastorm')

train_petastorm_converter = make_spark_converter(train_df)
val_petastorm_converter = make_spark_converter(val_df)

train_size = len(train_petastorm_converter)
val_size = len(val_petastorm_converter)

# COMMAND ----------

# MAGIC %md
# MAGIC ## iv. Define model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will define the same model as in [`02_model_training_single_node`]($./02_model_training_single_node), along with the same preprocessing functionality.

# COMMAND ----------

def build_model(img_height: int, 
                img_width: int,                 
                img_channels: int, 
                num_classes: int) -> tf.keras.models.Sequential:
  
    base_model = tf.keras.applications.MobileNetV2(include_top=False, 
                                                   input_shape=(img_height, img_width, img_channels))

    #freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model, 
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
        ])

    return model

# COMMAND ----------

def preprocess(content: str, label_idx: int):
    """
    Preprocess an image file bytes for MobileNetV2 (ImageNet).
    """
    image = tf.image.decode_jpeg(content, channels=IMG_CHANNELS)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    return preprocess_input(image), label_idx

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## v. Single Node - Feed data to single node using Petastorm
# MAGIC 
# MAGIC We will first do single node training where we will use Petastorm to feed the data from Delta to the training process. We need to use Petastorm's <a href="https://petastorm.readthedocs.io/en/latest/api.html#petastorm.spark.spark_dataset_converter.SparkDatasetConverter.make_tf_dataset" target="_blank">make_tf_dataset</a> to read batches of data.<br><br>
# MAGIC 
# MAGIC * Note that we use **`num_epochs=None`** to generate infinite batches of data to avoid handling the last incomplete batch. This is particularly useful in the distributed training scenario, where we need to guarantee that the numbers of data records seen on all workers are identical. Given that the length of each data shard may not be identical, setting **`num_epochs`** to any specific number would fail to meet the guarantee.
# MAGIC * The **`workers_count`** param specifies the number of threads or processes to be spawned in the reader pool, and it is not a Spark worker. 

# COMMAND ----------

with train_petastorm_converter.make_tf_dataset(batch_size=BATCH_SIZE) as train_ds,\
     val_petastorm_converter.make_tf_dataset(batch_size=BATCH_SIZE) as val_ds:

    # mlflow autologging
    mlflow.tensorflow.autolog()

    # Transforming datasets to map our preprocess function() and then batch()
    train_ds = train_ds.unbatch().map(lambda x: (x.content, x.label_idx))
    val_ds = val_ds.unbatch().map(lambda x: (x.content, x.label_idx))

    train_ds = (train_ds
                .map(lambda content, label_idx: preprocess(content, label_idx))
                .batch(BATCH_SIZE))
    val_ds = (val_ds
              .map(lambda content, label_idx: preprocess(content, label_idx))
              .batch(BATCH_SIZE))

    model = build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=EPOCHS,
              verbose=1,
              validation_data=val_ds,
              validation_steps=validation_steps)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see the code to train our model is exactly the same as our single node training. Adding Petastorm to load our data directly from the underlying parquet files in our delta tables simply required using the `make_spark_converter` class and initializing our datasets using the `make_tf_dataset()` method.
# MAGIC 
# MAGIC Next, we need to distribute the training across our cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## v. Distribute training with Horovod
# MAGIC 
# MAGIC [Horovod](https://github.com/horovod/horovod) is a distributed training framework for TensorFlow, Keras, and PyTorch. Databricks supports distributed deep learning training using HorovodRunner and the `horovod.spark` package which we will use next. 
# MAGIC 
# MAGIC We can use Horovod to train across multiple machines, meaning we can distribute training across CPU-clusters or GPU clusters. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC [HorovodRunner](https://databricks.github.io/spark-deep-learning/#sparkdl.HorovodRunner) is a general API to run distributed DL workloads on Databricks using Uber’s <a href="https://github.com/uber/horovod" target="_blank">Horovod</a> framework. By integrating Horovod with Spark’s barrier mode, Databricks is able to provide higher stability for long-running deep learning training jobs on Spark. 
# MAGIC 
# MAGIC ### How it works
# MAGIC * HorovodRunner takes a Python method that contains DL training code with Horovod hooks. 
# MAGIC * This method gets pickled on the driver and sent to Spark workers. 
# MAGIC * A Horovod MPI job is embedded as a Spark job using barrier execution mode. 
# MAGIC * The first executor collects the IP addresses of all task executors using BarrierTaskContext and triggers a Horovod job using mpirun. 
# MAGIC * Each Python MPI process loads the pickled program back, deserializes it, and runs it.
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/horovod-runner.png)
# MAGIC 
# MAGIC For additional resources, see:
# MAGIC * <a href="https://docs.microsoft.com/en-us/azure/databricks/applications/deep-learning/distributed-training/horovod-runner" target="_blank">Horovod Runner Docs</a>
# MAGIC * <a href="https://vimeo.com/316872704/e79235f62c" target="_blank">Horovod Runner webinar</a>  

# COMMAND ----------

# MAGIC %md
# MAGIC Inside our function, we'll incorporate the same logic we used in the single node and petastorm training examples plus additional logic to account for GPUs. 
# MAGIC 
# MAGIC If you have a CPU cluster, then these lines will be ignored and training will occur on CPU.

# COMMAND ----------

def train_and_evaluate_hvd():
    hvd.init()  # Initialize Horovod.
  
    # To enable tracking from the Spark workers to Databricks
    mlflow.mlflow.set_tracking_uri('databricks')
    os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
    os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
  
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
      
    # Creating model
    model = build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_classes)
  
    # Horovod: adjust learning rate based on number of workers.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
    dist_optimizer = hvd.DistributedOptimizer(optimizer)
  
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=0.001*hvd.size(), warmup_epochs=5, verbose=1),

        # Reduce the learning rate if training plateaus.
        ReduceLROnPlateau(patience=10, verbose=1)
    ]

    # Set experimental_run_tf_function=False in TF 2.x
    model.compile(optimizer=dist_optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    
    
    # same logic as Petastorm training example here with slight adjustments
    with train_petastorm_converter.make_tf_dataset(batch_size=BATCH_SIZE, 
                                         cur_shard=hvd.rank(), 
                                         shard_count=hvd.size()) as train_ds,\
        val_petastorm_converter.make_tf_dataset(batch_size=BATCH_SIZE, 
                                      cur_shard=hvd.rank(), 
                                      shard_count=hvd.size()) as val_ds:

        # Transforming datasets to map our preprocess function() and then batch()
        train_ds = train_ds.unbatch().map(lambda x: (x.content, x.label_idx))
        val_ds = val_ds.unbatch().map(lambda x: (x.content, x.label_idx))
    
        train_ds = (train_ds
                    .map(lambda content, label_idx: preprocess(content, label_idx))
                    .batch(BATCH_SIZE))
        val_ds = (val_ds
                  .map(lambda content, label_idx: preprocess(content, label_idx))
                  .batch(BATCH_SIZE))

        steps_per_epoch = train_size // (BATCH_SIZE * hvd.size())
        validation_steps = max(1, val_size // (BATCH_SIZE * hvd.size()))

        hist = model.fit(train_ds,
                         steps_per_epoch = steps_per_epoch,
                         epochs = EPOCHS,
                         verbose = 1,
                         validation_data = val_ds,
                         validation_steps = validation_steps)
        
        # MLflow Tracking (Log only from Worker 0)
        if hvd.rank() == 0:
            # Log events to MLflow
            with mlflow.start_run(run_id = active_run_uuid):
                # Log MLflow Parameters
                mlflow.log_param('epochs', EPOCHS)
                mlflow.log_param('batch_size', BATCH_SIZE)

                # Log MLflow Metrics
                mlflow.log_metric('val_loss', hist.history['val_loss'][-1])
                mlflow.log_metric('val_accuracy', hist.history['val_accuracy'][-1])

                # Log Model
                mlflow.keras.log_model(model, 'model')

    return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Training on the driver with Horovod

# COMMAND ----------

# MAGIC %md Test it out on just the driver.
# MAGIC 
# MAGIC `np=-1` will force Horovod to run on a single core on the Driver node.

# COMMAND ----------

with mlflow.start_run(run_name='horovod_driver') as run:
    
    active_run_uuid = mlflow.active_run().info.run_uuid
    hr = HorovodRunner(np=-1, driver_log_verbosity="all")
    hr.run(train_and_evaluate_hvd)
    
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distributed Training with Horovod

# COMMAND ----------

## OPTIONAL: You can enable Horovod Timeline as follows, but can incur slow down from frequent writes, and have to export out of Databricks to upload to chrome://tracing
# import os
# os.environ["HOROVOD_TIMELINE"] = f"{working_dir}/_timeline.json"

with mlflow.start_run(run_name='horovod_distributed') as run:
    
    active_run_uuid = mlflow.active_run().info.run_uuid
    hr = HorovodRunner(np=2, driver_log_verbosity='all')  
    hr.run(train_and_evaluate_hvd)
    
mlflow.end_run()

# COMMAND ----------

# MAGIC %md Finally, we <a href="https://petastorm.readthedocs.io/en/latest/api.html#petastorm.spark.spark_dataset_converter.SparkDatasetConverter.delete" target="_blank">delete</a> the cached Petastorm files.

# COMMAND ----------

train_petastorm_converter.delete()
val_petastorm_converter.delete()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## vi. Load model
# MAGIC 
# MAGIC Load model from MLflow

# COMMAND ----------

trained_model = mlflow.keras.load_model(f'runs:/{run.info.run_id}/model')
trained_model.summary()
