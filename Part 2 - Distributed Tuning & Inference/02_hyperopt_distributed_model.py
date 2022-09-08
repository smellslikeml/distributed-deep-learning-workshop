# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Tune distributed models with Hyperopt and Apache Spark

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook:
# MAGIC * We perform hyperparameter optimization of a distributed DL training process using [Hyperopt](https://github.com/hyperopt/hyperopt)
# MAGIC * We examine how to get the best performing model via the [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) API
# MAGIC * We subsequently register the best performing model to [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
import time
from typing import Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

from petastorm.spark import SparkDatasetConverter, make_spark_converter

from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, STATUS_OK, SparkTrials

import mlflow
from mlflow.tracking import MlflowClient

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
NUM_CLASSES = 5

BATCH_SIZE = 32
EPOCHS = 3

# Define directory to save model checkpoints to
checkpoint_dir = f'/dbfs/distributed_dl_workshop_{user}/train_ckpts/{time.time()}'
dbutils.fs.mkdirs(checkpoint_dir.replace('/dbfs', 'dbfs:'))

# Number of Horovod processes
HVD_NUM_PROCESSES = 2

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ii. Load prepared data from Silver layer 

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

# MAGIC %md 
# MAGIC ## iii. Convert the Spark DataFrame to a TensorFlow Dataset

# COMMAND ----------

dbutils.fs.rm(f'/tmp/distributed_dl_workshop_{user}/petastorm', recurse=True)
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, f'file:///dbfs/tmp/distributed_dl_workshop_{user}/petastorm')

train_petastorm_converter = make_spark_converter(train_df)
val_petastorm_converter = make_spark_converter(val_df)

train_size = len(train_petastorm_converter)
val_size = len(val_petastorm_converter)

# COMMAND ----------

def build_model(dropout: int = 0.5) -> tf.keras.models.Sequential:

    base_model = tf.keras.applications.MobileNetV2(include_top=False, 
                                                   input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model, 
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(NUM_CLASSES)
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
# MAGIC ## iv. Define training function for HorovodRunner
# MAGIC 
# MAGIC The following creates the training function. This function:
# MAGIC - Takes as it's arguments the hyperparameters for tuning
# MAGIC - Configures Horovod
# MAGIC - Loads the training and validation datasets using Petastorm
# MAGIC - Defines the model and distributed optimizer
# MAGIC - Compiles the model
# MAGIC - Fits the model
# MAGIC - Returns the loss and accuracy on the validation set 

# COMMAND ----------

def train_and_evaluate_hvd(learning_rate: float,
                           dropout: float,
                           batch_size: int,
                           checkpoint_dir: str) -> Tuple[float, float]:
    """
    Function to pass to Horovod to be executed on each worker.
    Args are those hyperparameters we want to tune with Hyperopt.
    """
    # Initialize Horovod.
    hvd.init()  
  
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
    model = build_model(dropout=dropout)
  
    # Horovod: adjust learning rate based on number of workers.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate * hvd.size())
    dist_optimizer = hvd.DistributedOptimizer(optimizer)
  
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=learning_rate*hvd.size(), 
                                                 warmup_epochs=5, 
                                                 verbose=1),
        # Reduce the learning rate if training plateaus.
        ReduceLROnPlateau(patience=10, verbose=1)
    ]

    model.compile(optimizer=dist_optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    
    # Save checkpoints only on worker 0 to prevent conflicts between workers
    param_str = f'learning_rate_{learning_rate}_dropout_{dropout}_batch_size_{batch_size}'
    if hvd.rank() == 0:        
        checkpoint_dir_for_this_trial = os.path.join(checkpoint_dir, param_str)
        local_ckpt_path = os.path.join(checkpoint_dir_for_this_trial, 'checkpoint-{epoch}.ckpt')
        callbacks.append(ModelCheckpoint(local_ckpt_path, save_weights_only=True))

    with train_petastorm_converter.make_tf_dataset(batch_size=batch_size, 
                                         cur_shard=hvd.rank(), 
                                         shard_count=hvd.size()) as train_ds,\
        val_petastorm_converter.make_tf_dataset(batch_size=batch_size, 
                                      cur_shard=hvd.rank(), 
                                      shard_count=hvd.size()) as val_ds:

        train_ds = train_ds.unbatch().map(lambda x: (x.content, x.label_idx))
        val_ds = val_ds.unbatch().map(lambda x: (x.content, x.label_idx))
    
        train_ds = (train_ds
                    .map(lambda content, label_idx: preprocess(content, label_idx))
                    .batch(batch_size))
        val_ds = (val_ds
                  .map(lambda content, label_idx: preprocess(content, label_idx))
                  .batch(batch_size))

        steps_per_epoch = train_size // (batch_size * hvd.size())
        validation_steps = max(1, val_size // (batch_size * hvd.size()))

        hist = model.fit(train_ds,
                         steps_per_epoch=steps_per_epoch,
                         epochs=EPOCHS,
                         verbose=1,
                         validation_data=val_ds,
                         validation_steps=validation_steps)
                
        # MLflow Tracking (Log only from Worker 0)
        if hvd.rank() == 0:
            # Log to child run under active parent run
            # MLFLOW_PARENT_RUN_ID is defined below, prior to kicking of the HPO
            with mlflow.start_run(run_id=MLFLOW_PARENT_RUN_ID) as run:
                with mlflow.start_run(experiment_id=run.info.experiment_id,
                                      run_name=param_str, 
                                      nested=True):
                    # Log MLflow Parameters
                    mlflow.log_param('epochs', EPOCHS)
                    mlflow.log_param('batch_size', batch_size)
                    mlflow.log_param('learning_rate', learning_rate)
                    mlflow.log_param('dropout', dropout)
                    mlflow.log_param('checkpoint_dir', checkpoint_dir_for_this_trial)

                    # Log MLflow Metrics
                    mlflow.log_metric('val_loss', hist.history['val_loss'][-1])
                    mlflow.log_metric('val_accuracy', hist.history['val_accuracy'][-1])

                    # Log Model
                    mlflow.keras.log_model(model, 'model')

    return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## iii. Hyperopt Workflow
# MAGIC 
# MAGIC In the following section we create the Hyperopt workflow to tune our distributed DL training process. We will:
# MAGIC * Define an object function to minimize
# MAGIC * Define a search space over hyperparameters
# MAGIC * Specify the search algorithm and use `fmin()` to tune the model
# MAGIC 
# MAGIC For more information about the Hyperopt APIs, see the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define objective function
# MAGIC 
# MAGIC First, we need to [create an **objective function**](http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/). This is the function that Hyperopt will call for each set of inputs.
# MAGIC 
# MAGIC The basic requirements are:
# MAGIC 
# MAGIC 1. An **input** `params` including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize
# MAGIC 
# MAGIC In this case, we are specifying values of `learning_rate`, `dropout` and `batch_size` and returning validation accuracy as our loss metric.
# MAGIC 
# MAGIC Our objective function will use the training function defined for `HorovodRunner` and run distributed training to fit a model and compute its loss. To run this example on a cluster with 2 workers, each with a single GPU, initialize `HorovodRunner` with `np=2`.

# COMMAND ----------

def objective_function(params):
    """
    An example train method that calls into HorovodRunner.
    This method is passed to hyperopt.fmin().

    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    hr = HorovodRunner(np=HVD_NUM_PROCESSES)
    loss, acc = hr.run(train_and_evaluate_hvd,
                       learning_rate=params['learning_rate'],
                       dropout=params['dropout'],
                       batch_size=params['batch_size'],
                       checkpoint_dir=checkpoint_dir)
    
    return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the search space
# MAGIC 
# MAGIC Next, we need to [define the **search space**](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/).
# MAGIC 
# MAGIC This example tunes 3 hyperparameters: learning rate, dropout and batch size. See the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for details on defining a search space and parameter expressions.

# COMMAND ----------

search_space = {
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'dropout': hp.uniform('dropout', 0.1, 0.9),
    'batch_size': hp.choice('batch_size', [32, 64, 128])
}

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Tune the model using Hyperopt `fmin()`
# MAGIC 
# MAGIC - Set `max_evals` to the maximum number of points in hyperparameter space to test, that is, the maximum number of models to fit and evaluate. Because this command evaluates many models, it will take several minutes to execute.
# MAGIC - You must also specify which search algorithm to use. The two main choices are:
# MAGIC   - `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# MAGIC   - `hyperopt.rand.suggest`: Random search, a non-adaptive approach that randomly samples the search space

# COMMAND ----------

# MAGIC %md 
# MAGIC **Important:** We are training trials (each distributed training process) in a *sequential* manner. As such, when using Hyperopt with HorovodRunner, do not pass a `trials` argument to `fmin()`. 
# MAGIC 
# MAGIC When you do not include the `trials` argument, Hyperopt uses the default `Trials` class, which runs on the cluster driver. Hyperopt needs to evaluate each trial on the driver node so that each trial can initiate distributed training jobs.  The `SparkTrials` class is incompatible with distributed training jobs, as it evaluates each trial on one worker node.

# COMMAND ----------

# Start a parent MLflow run
mlflow_exp_path = '/Users/' + user + '/distributed_dl_workshop'
mlflow.set_experiment(mlflow_exp_path)

with mlflow.start_run(run_name='hyperopt_horovod_tuning') as hyperopt_mlflow_run:
    
    MLFLOW_PARENT_RUN_ID = hyperopt_mlflow_run.info.run_id
    
    # The number of models we want to evaluate
    max_evals = 4

    # Run the optimization process
    best_hyperparam = fmin(
        fn=objective_function, 
        space=search_space,
        algo=tpe.suggest, 
        max_evals=max_evals,
    )

    # Log optimal hyperparameter values
    mlflow.log_param('learning_rate', best_hyperparam['learning_rate'])
    mlflow.log_param('dropout', best_hyperparam['dropout'])
    mlflow.log_param('batch_size', best_hyperparam['batch_size'])    

# COMMAND ----------

# Print out the parameters that produced the best model
print(best_hyperparam)

# COMMAND ----------

# Display the contents of the checkpoint directory
dbutils.fs.ls(checkpoint_dir.replace('/dbfs', 'dbfs:'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iv. Get the best run via MLflow Tracking

# COMMAND ----------

# Get the run_id of the parent run under which the Hyperopt trials were tracked
parent_run_id = hyperopt_mlflow_run.info.run_id

# Return all trials (tracked as child runs) as a pandas DataFrame, and order by accuracy descending
hyperopt_trials_pdf = (mlflow.search_runs(filter_string=f'tags.mlflow.parentRunId="{parent_run_id}"', 
                                          order_by=['metrics.accuracy DESC']))

# The best trial will be the first row of this pandas DataFrame
best_run = hyperopt_trials_pdf.iloc[0]
best_run_id = best_run['run_id']

print(f'MLflow run_id of best trial: {best_run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## v. Register best run to MLfow Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We register our model to the MLflow Model Registry to use in our subsequent inference notebook.

# COMMAND ----------

# Define unique name for model in the Model Registry
registry_model_name = my_name + '_flower_classifier'

# Register the best run. Note that when we initially register it, the model will be in stage=None
model_version = mlflow.register_model(f'runs:/{best_run_id}/model', 
                                      name=registry_model_name)

# COMMAND ----------

# Transition model to stage='Production'
client = MlflowClient()
client.transition_model_version_stage(
    name=registry_model_name,
    version=model_version.version,
    stage='Production'
)
