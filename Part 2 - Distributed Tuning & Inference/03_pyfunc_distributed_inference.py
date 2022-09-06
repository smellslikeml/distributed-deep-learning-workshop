# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Packaging an MLflow PyFunc, and distributed inference

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this notebook we examine how to use our trained model to perform inference in a distributed manner. We will do so wrapping our `model.predict()` method as a PySpark UDF.
# MAGIC 
# MAGIC In order to accomplish this we will package our image preprocessing and model prediction functionality as an [MLflow pyfunc model](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html). Packaging our model in this format to enable this capability requires that we do so at model training time.
# MAGIC 
# MAGIC Thus, in this notebook we will:
# MAGIC 
# MAGIC - Define functions to enable model training, and logging model as a [pyfunc model flavour](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html).
# MAGIC   - Training will be executed on a single (driver) node, using Petastorm to feed data to the training process.
# MAGIC   - Model parameters, metrics and artifacts will be tracked to MLflow.
# MAGIC   - Inference logic will be packaged into the logged MLflow pyfunc model.
# MAGIC - Execute model training on a single node.
# MAGIC - Load our model for inference.
# MAGIC   - First testing against a pandas DataFrame in a single node setting.
# MAGIC   - Then applying our model against a Spark DataFrame in a distributed manner.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import io
import os
import json
import ast
from typing import Tuple
from PIL import Image

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import mlflow

from petastorm.spark import SparkDatasetConverter, make_spark_converter

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Set global variables

# COMMAND ----------

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

BATCH_SIZE = 128
EPOCHS = 3

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training functions
# MAGIC 
# MAGIC Before executing the training process, we define the various classes and functions to enable this.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Define dataclass for data config at training time
# MAGIC 
# MAGIC We use the following `dataclass`[docs](https://docs.python.org/3/library/dataclasses.html) to define the data inputs and cache directory for our training process. This user-defined class will be define prior to passing into the main train function.

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class DataCfg:
    """
    Class representing data inputs to the training process
    """
    train_tbl_name: str
    validation_tbl_name: str
    petastorm_cache_dir: str = f'/tmp/distributed_dl_workshop_{user}/petastorm'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Define preprocessing to decode and resize images
# MAGIC 
# MAGIC The following `preprocess` function is used to decode and resize the binary image inputs in the training pipeline. Note that these same preprocessing steps must be applied to raw binary image inputs at inference time. We will see below how we can package this up as a [custom MLflow pyfunc model](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models).

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
# MAGIC ### Define model architecture
# MAGIC 
# MAGIC We define the same model architecture as that used throughout the rest of the examples.

# COMMAND ----------

def build_model() -> tf.keras.models.Sequential:
    """
    Function to define model architecture
    """  
    base_model = tf.keras.applications.MobileNetV2(include_top=False, 
                                                   input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model, 
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASSES))
        ])

    return model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Define custom MLflow PyFunc
# MAGIC 
# MAGIC Beforehand we saw how we could train and track our `tf.keras` to MLflow. That logged model artifact was simply the trained model artifact. In order to use that model at inference time we must apply the same preprocessing steps, and additional would have to translate our model output to predicted class labels.
# MAGIC 
# MAGIC We can leverage custom Pyfunc models in MLflow to package our custom preprocessing and postprocessing routines. When we load the model at inference time the model artifacts declared in the `load_context` method are loaded, and the custom inference logic within the `predict` method is applied to the inference dataset.

# COMMAND ----------

class FlowerPyFunc(mlflow.pyfunc.PythonModel):
    """
    Custom PyFunc class to enable prediction with tf.keras model against a binary encoded images
    """
    def load_context(self, context):
        """
        Loads artifacts to be used by `PythonModel.predict` when evaluating inputs.
        When loading an MLflow model with `.load_model`, this method is called as soon as the 
        `PythonModel` is constructed.
        
        Args:
            context (`mlflow.pyfunc.PythonModelContext`):
                A `PythonModelContext` instance containing artifacts that the model can use to 
                perform inference.
        """
        # Path to logged image params dict
        img_params_dict_path = context.artifacts['img_params_dict_path']        
        with open(img_params_dict_path) as f:
            img_params_dict = json.load(f)
            
        # Set image params
        self.img_height = img_params_dict['img_height']
        self.img_width = img_params_dict['img_width']
        
        # Path to logged model
        keras_model_path = context.artifacts['keras_model_path'] 
        # Load Keras model
        self.model = mlflow.keras.load_model(keras_model_path)

    def predict(self, context, model_input: pd.Series) -> np.array:
        """
        Take a pd.Series of binary encoded images (BinaryType 'contents' colummn from Binary file data source),
        preprocess images, and apply model

        See https://docs.databricks.com/data/data-sources/binary-file.html#binary-file for info on data source
                
        Args:
            context: 
                instance containing artifacts that the model can use to perform inference
            model_input (pd.Series): 
                pd.Series of binary encoded images
                
        Returns: 
            np.array of predicted class labels 
        """
        model_input_np = np.array(model_input)
        # Preprocess binary images
        preprocessed_arr = np.array(list(map(self.preprocess, model_input_np)))        
        # Predictions
        pred_arr = self.model.predict(preprocessed_arr, batch_size=BATCH_SIZE)        
        # Predicted indices
        pred_idx = np.argmax(pred_arr, axis=1)
        # Predicted class labels
        pred_classes = np.take(CLASSES, pred_idx, axis=0)

        return pred_classes      
  
    def preprocess(self, img_bytes: bytes) -> np.array:   
        """
        Take a single binary encoded image, decode image to numpy array
        and resize according to image height and size set in config.
        
        Args:
            img_bytes (bytes): 
                A single binary encoded image.
                
        Returns: 
            np.array of shape [img_height, img_width].
        """
        # When pyfunc is applied as SparkUDF byte format is converted to str
        # We require bytes format        
        if type(img_bytes) is not bytes:
            img_bytes = ast.literal_eval(str(img_bytes[0]))
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize([self.img_height, self.img_width])

        return np.asarray(img, dtype='float32')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Define training function
# MAGIC 
# MAGIC The following function encapsulates the full training pipeline:
# MAGIC 
# MAGIC - Creating the MLflow run to log parameters, metrics and artifacts to
# MAGIC - Setting up the Petastorm converter and data access process
# MAGIC - Defining, compiling and fitting the model (single node training)
# MAGIC - Logging the custom Pyfunc model with packaged model inference logic
# MAGIC - Evaluating model performance against the validation dataset
# MAGIC - Returning the MLflow run object, along with model and model history object

# COMMAND ----------

def train_model_petastorm_data_ingest(data_cfg: DataCfg,
                                      compile_kwargs: dict,
                                      fit_kwargs: dict) -> Tuple[mlflow.tracking.fluent.Run,
                                                                 tf.keras.Model,
                                                                 tf.keras.callbacks.History]:
    """
    Function to train a tf.keras model as defined by `build_model()`, on a single node.
    Data is fed to the training process using Petastorm, with all params, metrics and model artifacts
    logged to MLflow. The Petastorm converter objects are deleted upon completion of model evaluation.
    
    Args:
        data_cfg (DataCfg):
            Data class representing data inputs to the training process
        compile_kwargs (dict):
            Compile kwargs passed to model.compile()
        fit_kwargs (dict):
            Model fit kwargs passed to model.fit(). Note that the batch_size defined in this dict is used
            to define the Petastorm data access process rather than explicitly passed into model.fit().
            
    Returns:
        Tuple of MLflow Run object (mlflow.tracking.fluent.Run), trained tf.keras model (tf.keras.Model),
        and fitted model history object (tf.keras.callbacks.History)
    """
    # Enable autologging
    mlflow.autolog()
    
    with mlflow.start_run(run_name='pyfunc_model_petastorm') as mlflow_run:
        
        #####################
        # LOG RUN PARAMS
        #####################        
        # Log img_height and img_width for use in PyFunc (required for preprocessing)
        img_params_dict = {'img_height': IMG_HEIGHT, 
                           'img_width': IMG_WIDTH}
        mlflow.log_dict(img_params_dict, 'img_params_dict.json')
        mlflow.log_param('BATCH_SIZE', fit_kwargs['batch_size'])

        #####################
        # DATA LOADING
        #####################
        print('DATA LOADING')    
        # Load and repartition train and validation datasets
        # Hard-coding the number of partitions. Should be at least the number of workers which is required for distributed training.
        train_df = spark.table(data_cfg.train_tbl_name).select('content', 'label_idx').repartition(2)
        val_df = spark.table(data_cfg.validation_tbl_name).select('content', 'label_idx').repartition(2)

        # Set directory to use for the Petastorm cache
        dbutils.fs.rm(data_cfg.petastorm_cache_dir, recurse=True)
        spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, f'file:///dbfs{data_cfg.petastorm_cache_dir}')

        # Create Petastorm Spark Converters for train and validation datasets
        print('Creating Petastorm Spark Converters')            
        train_petastorm_converter = make_spark_converter(train_df)
        val_petastorm_converter = make_spark_converter(val_df)
        train_size = len(train_petastorm_converter)
        val_size = len(val_petastorm_converter)

        with train_petastorm_converter.make_tf_dataset(batch_size=fit_kwargs['batch_size']) as train_ds,\
             val_petastorm_converter.make_tf_dataset(batch_size=fit_kwargs['batch_size']) as val_ds:

            #####################
            # DATA PREPROCESSING
            #####################   
            print('DATA PREPROCESSING')
            # Transforming datasets to map our preprocess function() and then batch()
            train_ds = train_ds.unbatch().map(lambda x: (x.content, x.label_idx))
            val_ds = val_ds.unbatch().map(lambda x: (x.content, x.label_idx))

            train_ds = (train_ds
                        .map(lambda content, label_idx: preprocess(content, label_idx))
                        .batch(fit_kwargs['batch_size']))
            val_ds = (val_ds
                      .map(lambda content, label_idx: preprocess(content, label_idx))
                      .batch(fit_kwargs['batch_size']))

            #####################
            # MODEL TRAINING
            ##################### 
            print('MODEL TRAINING')                      
            model = build_model()
            model.compile(**compile_kwargs)

            steps_per_epoch = train_size // fit_kwargs['batch_size']
            validation_steps = val_size // fit_kwargs['batch_size']

            history = model.fit(train_ds,
                                steps_per_epoch=steps_per_epoch,
                                epochs=fit_kwargs['epochs'],
                                verbose=fit_kwargs['verbose'],
                                validation_data=val_ds,
                                validation_steps=validation_steps)
            print('MODEL FIT COMPLETE')                                            

            #####################
            # PyFunc
            #####################
            # Image params dict
            img_params_dict_path = f'runs:/{mlflow_run.info.run_id}/img_params_dict.json'
            # Keras model
            keras_model_path = f'runs:/{mlflow_run.info.run_id}/model'         

            pyfunc_artifacts = {
                                'img_params_dict_path': img_params_dict_path,
                                'keras_model_path': keras_model_path,
                                }
            # Log MLflow pyfunc Model
            print('LOGGING PYFUNC')                                                                  
            mlflow.pyfunc.log_model('pyfunc_model', 
                                    python_model=FlowerPyFunc(), 
                                    artifacts=pyfunc_artifacts
                                   )             
            
            #####################
            # MODEL EVALUATION
            #####################
            print('MODEL EVALUATION')                                                                                        
            val_results = model.evaluate(val_ds, steps=validation_steps)
            val_results_dict = dict(zip(model.metrics_names, val_results))
            mlflow.log_metrics({f'val_{metric}':value for metric, value in val_results_dict.items()})   
            
    print('DELETING PETASTORM CONVERTER')
    train_petastorm_converter.delete()
    val_petastorm_converter.delete()            

    return mlflow_run, model, history

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Model training
# MAGIC 
# MAGIC Let's now define our training data config class, along with our compile and fit arguments and trigger model training.

# COMMAND ----------

data_cfg = DataCfg(train_tbl_name=f'{database_name}.silver_train',
                   validation_tbl_name=f'{database_name}.silver_val')

compile_kwargs = {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
                  'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  'metrics': ['accuracy']}

# Define fit args
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  min_delta=1e-2, 
                                                  patience=3, 
                                                  verbose=1)
callbacks = [early_stopping]
fit_kwargs = {'batch_size': BATCH_SIZE,
              'epochs': 2,    # Run for 2 epochs for demonstration purposes
              'verbose': 1}


mlflow_run, _, _ = train_model_petastorm_data_ingest(data_cfg=data_cfg,
                                                     compile_kwargs=compile_kwargs,
                                                     fit_kwargs=fit_kwargs)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Inference
# MAGIC 
# MAGIC We now look at applying our model to some inference data. For demo purposes we are using the `silver` table created earlier.

# COMMAND ----------

# Use the silver table as our inference table for demo purposes
inference_df = spark.table(f'{database_name}.silver')

print('inference_df count:', inference_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Single node inference
# MAGIC 
# MAGIC We can first look at applying our model in a single node setting for inference.
# MAGIC 
# MAGIC - Define a model_uri from which to load the Pyfunc model
# MAGIC - Apply loaded model against a pandas DataFrame

# COMMAND ----------

# Get run_id from the run object executed above
run_id = mlflow_run.info.run_id
model_uri = f'runs:/{run_id}/pyfunc_model'
print(model_uri)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri)
inference_pdf = inference_df.limit(10).toPandas()
inference_pred_arr = loaded_model.predict(inference_pdf['content'])

inference_pred_arr

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distributed inference
# MAGIC 
# MAGIC We can use the same Pyfunc model in a distributed setting. To do so we:
# MAGIC - Define a model_uri from which to load the Pyfunc model
# MAGIC - Create a PySpark UDF that can be used to invoke our custom Pyfunc model
# MAGIC   - Note that our `result_type` will be a string in this example
# MAGIC - Apply the UDF to the `content` column of the PySpark Dataframe. This column contains the raw binary representation of our images

# COMMAND ----------

loaded_model_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type='string')

inference_pred_df = (inference_df
                     .limit(1000)
                     .withColumn('prediction', loaded_model_udf('content'))
                     .select('path', 'content', 'label', 'prediction')
                    )

# COMMAND ----------

inference_pred_df.display()
