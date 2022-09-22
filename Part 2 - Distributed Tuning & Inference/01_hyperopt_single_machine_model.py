# Databricks notebook source
# MAGIC %md
# MAGIC # 01. Tune Single-node Models with Hyperopt and Apache Spark

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook:
# MAGIC * We perform hyperparameter optimization, training multiple single node models in parallel using [Hyperopt](https://github.com/hyperopt/hyperopt)
# MAGIC * We examine how to get the best performing model via the [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) API
# MAGIC * We subsequently register the best performing model to [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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


# Convert Spark DataFrame to pandas DataFrame
train_pdf = train_df.toPandas() 
val_pdf = val_df.toPandas()

# COMMAND ----------

def preprocess(content: str, label_idx: int):
    """
    Preprocess an image file bytes for MobileNetV2 (ImageNet).
    """
    image = tf.image.decode_jpeg(content, channels=IMG_CHANNELS)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    return preprocess_input(image), label_idx

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

# MAGIC %md
# MAGIC ## iii. Hyperopt Workflow
# MAGIC 
# MAGIC Next, we will create the different pieces needed for parallelizing hyperparameter tuning with [Hyperopt](http://hyperopt.github.io/hyperopt/) and Apache Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ### a. Create objective function
# MAGIC 
# MAGIC First, we need to [create an **objective function**](http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/). This is the function that Hyperopt will call for each set of inputs.
# MAGIC 
# MAGIC The basic requirements are:
# MAGIC 
# MAGIC 1. An **input** `params` including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize
# MAGIC 
# MAGIC In this case, we are specifying values of `optimizer`, `learning_rate` and `dropout` and returning validation accuracy as our loss metric.

# COMMAND ----------

def objective_function(params):
    
    mlflow.autolog()

    # Create train tf.data.Dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (train_pdf['content'].values, train_pdf['label_idx'].values))
        .map(lambda content, label_idx: preprocess(content, label_idx))
        .batch(BATCH_SIZE)
    )

    # Create val tf.data.Dataset
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (val_pdf['content'].values, val_pdf['label_idx'].values))
        .map(lambda content, label_idx: preprocess(content, label_idx))
        .batch(BATCH_SIZE)        
    )

    # Select Optimizer
    optimizer_call = getattr(tf.keras.optimizers, params['optimizer'])
    optimizer = optimizer_call(learning_rate=params['learning_rate'])

    # Instantiate model
    model = build_model(dropout=params['dropout'])

    # Compile model
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Specify training params
    steps_per_epoch = len(train_ds) // BATCH_SIZE
    validation_steps = len(val_ds) // BATCH_SIZE

    # Train the model
    model.fit(train_ds,
              steps_per_epoch = steps_per_epoch,
              epochs = EPOCHS,
              verbose = 1,
              validation_data = val_ds,
              validation_steps = validation_steps)
    
    _, accuracy = model.evaluate(val_ds)
    
    # Hyperopt will optimize to **minimize** the returned 'loss'
    # Given that we want to **maximize** accuracy, we return -accuracy
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ### b. Define the search space
# MAGIC 
# MAGIC Next, we need to [define the **search space**](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/).
# MAGIC 
# MAGIC Note that we aren't defining the actual values like grid search. Hyperopt's TPE algorithm will intelligently suggest hyperparameter values from within this range.

# COMMAND ----------

search_space = {
    'optimizer': hp.choice('optimizer', ['Adadelta', 'Adam']),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'dropout': hp.uniform('dropout', 0.1, 0.9)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### c. Call the `fmin` operation
# MAGIC 
# MAGIC The `fmin` function is where we put Hyperopt to work.
# MAGIC 
# MAGIC To make this work, we need:
# MAGIC 
# MAGIC 1. The `objective_function`
# MAGIC 2. The `search_space`
# MAGIC 3. The `tpe.suggest` optimization algorithm
# MAGIC 4. A `SparkTrials` object to distribute the trials across a cluster using Spark
# MAGIC 5. The maximum number of evaluations or trials denoted by `max_evals`
# MAGIC 
# MAGIC In this case, we'll be computing up to 20 trials with 4 trials being run concurrently.
# MAGIC 
# MAGIC In Databricks Hyperopt automatically logs its trials to MLflow under a single parent run, with each trial logged to a child run. Within the objective function we are using MLflow autologging which will result in the parameters, metrics and model artifacts logged to each child run.

# COMMAND ----------

mlflow.set_experiment('/Users/' + user + '/distributed_dl_workshop')

with mlflow.start_run(run_name='hyperopt_tuning') as hyperopt_mlflow_run:
    
    # The number of models we want to evaluate
    num_evals = 20

    # Set the number of models to be trained concurrently
    spark_trials = SparkTrials(parallelism=4)

    # Run the optimization process
    best_hyperparam = fmin(
        fn=objective_function, 
        space=search_space,
        algo=tpe.suggest, 
        trials=spark_trials,
        max_evals=num_evals
    )

    # Log optimal hyperparameter values
    mlflow.log_param('optimizer', best_hyperparam['optimizer'])
    mlflow.log_param('learning_rate', best_hyperparam['learning_rate'])
    mlflow.log_param('dropout', best_hyperparam['dropout'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iv. How to get the best run via MLflow Tracking

# COMMAND ----------

# Get the run_id of the parent run under which the Hyperopt trials were tracked
parent_run_id = hyperopt_mlflow_run.info.run_id

# Return all trials (tracked as child runs) as a pandas DataFrame, and order by accuracy descending
hyperopt_trials_pdf = (mlflow.search_runs(filter_string=f'tags.mlflow.parentRunId="{parent_run_id}"', 
                                          order_by=['metrics.accuracy DESC']))

# The best trial will be the first row of this pandas DataFrame
best_run = hyperopt_trials_pdf.iloc[0]
best_run_id = best_run['run_id']

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

# COMMAND ----------

# Load model from the production stage in MLflow Model Registry
prod_model = mlflow.keras.load_model(f'models:/{registry_model_name}/production')
prod_model.summary()
