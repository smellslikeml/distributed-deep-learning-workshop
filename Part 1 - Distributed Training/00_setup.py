# Databricks notebook source
# Move file from driver to DBFS
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# Parse name from user
my_name = user.split('@')[0].replace('.', '_')

# Set config for database name, file paths, and table names
database_name = f'distributed_dl_workshop_{my_name}'

print(f'database_name: {database_name}')

# COMMAND ----------

# In order for us to track from our worker nodes with Horovod to the MLflow Tracking server we need to supply a databricks host and token
DATABRICKS_HOST = 'https://' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
