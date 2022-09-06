# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # 04. Monitoring and Optimization

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
