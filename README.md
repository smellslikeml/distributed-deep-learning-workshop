# Distributed Deep Learning Workshop

In this workshop, we will train a deep learning model in a distributed manner using Databricks. We will discuss how we can leverage Delta Lake to prepare structured, semi-structured, or unstructured datasets and Petastorm for distributing datasets efficiently on a cluster. We will also cover how to use Horovod for distributed training on both CPU and GPU based hardware. This example aims to serve as a reusable template that is tailorable to meet your specific modeling needs.

## Workshop structure

The workshop involves a series of Databricks notebooks split into two parts,

In part 1 we look at how we can optimally leverage the parallelism of Spark for training deep learning models in a distributed manner. The notebooks outline the following:

- Data Prep
  - How to create a Delta table with the Binary file data source reader using JPEG image sources.
- Single node training
- Distributed training

In part 2 we look at how we can paralellize both hyperparameter tuning and model inference. We illustrate:

- Model tuning with Hyperopt
  - Tuning a single node DL model with Hyperopt
  - Tuning a distributed Horovod process with Hyperopt
- Distributed model inference
  - How to package up a custom Pyfunc with preprocessing/post-processing steps
  - Applying that logged custom Pyfunc in a single node inference setting
  - Applying that logged custom Pyfunc in a distributed inference setting

## Requirements

A recommended Databricks ML Runtime >= 7.3LTS is suggested. Please use the [repos](https://docs.databricks.com/repos/index.html) feature to clone into your repo and access the notebook.


