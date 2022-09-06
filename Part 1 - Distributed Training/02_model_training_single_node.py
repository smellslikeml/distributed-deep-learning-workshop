# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Model Training - Single node 

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook we will be performing some basic transfer learning on the flowers dataset, using the [`MobileNetV2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2) model as our base model.
# MAGIC 
# MAGIC We will:
# MAGIC - Load the train and validation datasets created in [01_data_prep]($./01_data_prep), converting them to `tf.data.Dataset` objects. 
# MAGIC - Train our model on a single (driver) node.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import mlflow

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

BATCH_SIZE = 32
EPOCHS = 3

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ii. Load prepared data from Silver layer 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Load the train and validation datasets we created in [01_data_prep]($./01_data_prep) as Spark DataFrames.

# COMMAND ----------

cols_to_keep = ['content', 'label_idx']

train_tbl_name = 'silver_train'
val_tbl_name = 'silver_val'

train_df = (spark.table(f'{database_name}.{train_tbl_name}')
            .select(cols_to_keep))

val_df = (spark.table(f'{database_name}.{val_tbl_name}')
          .select(cols_to_keep))

print('train_df count:', train_df.count())
print('val_df count:', val_df.count())

# COMMAND ----------

num_classes = train_df.select('label_idx').distinct().count()

print('Number of classes:', num_classes)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iii. Create `tf.data.Dataset`

# COMMAND ----------

# MAGIC %md
# MAGIC To train a model with Tensorflow, we'll need to transform our spark dataframes into `tf.data.Dataset` objects. Tensorflow has a function to transform a pandas dataframe into a [tensorflow dataset](https://www.tensorflow.org/datasets).
# MAGIC 
# MAGIC For this example, we'll convert our dataframe to a pandas dataframe using `.toPandas()` and convert to a properly formated `tf.Data.dataset` from there.

# COMMAND ----------

train_pdf = train_df.toPandas() 
val_pdf = val_df.toPandas()

# Create train tf.data.Dataset
train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (train_pdf['content'].values, train_pdf['label_idx'].values))
)

# Create val tf.data.Dataset
val_ds = (
    tf.data.Dataset.from_tensor_slices(
        (val_pdf['content'].values, val_pdf['label_idx'].values))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can continue transforming our image dataset into the correct format using a preprocessing function and mapping it to our `tf.datasets`

# COMMAND ----------

def preprocess(content: str, label_idx: int):
    """
    Preprocess an image file bytes for MobileNetV2 (ImageNet).
    """
    image = tf.image.decode_jpeg(content, channels=IMG_CHANNELS)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    return preprocess_input(image), label_idx

# COMMAND ----------

# Apply preprocess function to both train and validation datasets
train_ds = (train_ds
            .map(lambda content, label_idx: preprocess(content, label_idx))
            .batch(BATCH_SIZE)
           )

val_ds = (val_ds
          .map(lambda content, label_idx: preprocess(content, label_idx))
          .batch(BATCH_SIZE)
         )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## iv. Define model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Single node training is the most common way machine learning practitioners set up their training execution. For some modeling use cases, this is a great way to go as all the modeling can stay on one machine with no additional libraries. 
# MAGIC 
# MAGIC Training with Databricks is just as easy. Simply use a [Single Node Cluster](https://docs.databricks.com/clusters/single-node.html) which is just a Spark Driver with no worker nodes. 
# MAGIC 
# MAGIC We will be using the [`MobileNetV2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2) architecture from TensorFlow as our base model, freezing the base layers and adding a head of Global Pooling, Dropout and Dense classification layer.

# COMMAND ----------

def build_model(img_height: int,
                img_width: int,                  
                img_channels: int, 
                num_classes: int) -> tf.keras.models.Sequential:

    base_model = tf.keras.applications.MobileNetV2(include_top=False, 
                                                   input_shape=(img_height, img_width, img_channels))

    # Freeze base model layers
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

# MAGIC %md
# MAGIC 
# MAGIC ## v. Train Model - Single Node

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's train out model, where we will instantiate and compile the model, and fit the model; using [MLflow autologging](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/databricks-autologging) to automatically track model params, metrics and artifacts.

# COMMAND ----------

# mlflow autologging
mlflow.tensorflow.autolog()

# Instantiate model
model = build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_classes)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Specify training params
steps_per_epoch = len(train_ds) // BATCH_SIZE
validation_steps = len(val_ds) // BATCH_SIZE

# Train the model
model.fit(train_ds,
          steps_per_epoch=steps_per_epoch,
          epochs=EPOCHS,
          verbose=1,
          validation_data=val_ds,
          validation_steps=validation_steps)

# COMMAND ----------

# MAGIC %md
# MAGIC Wohoo! We trained a model on a single machine very similarly to what you might have done on your laptop. It was easy for us to convert this small dataset into a pandas DataFrame from a Delta table and subsequently convert it into a `tf.data.Dataset` for training. 
# MAGIC 
# MAGIC However, most production workloads will require training with **orders of magnitude** more data, so much so that it could overwhelm a single machine while training. Other cases could contribute to exhausting a single node during training such as training large model architectures.
# MAGIC 
# MAGIC In the next notebook, [03_model_training_distributed]($./03_model_training_distributed) we will see how we scale our model training across mutliple nodes.
