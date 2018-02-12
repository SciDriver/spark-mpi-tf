# SPARK-MPI-TF

This project demostrates the Spark-MPI approach within the context of Spark-based TensorFlow distributed deep learning
applications. The direction is addressed by several other projects, such as
[BigDL](https://github.com/intel-analytics/BigDL) and
[TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark). In comparison with these alternative
solutions, Spark-MPI aims to derive an application-neutral mechanism based on the MPI Process Management Interface (PMI)
for the effortless integration of Big Data and HPC ecosystems. 

## Prerequisites

1. [Spark-MPI](https://github.com/SciDriver/spark-mpi): PMI-based approach for integrating together the Spark platform and MPI applications

2. [Horovod](https://github.com/uber/horovod): MPI-based training framework for TensorFlow

## Examples

The MNIST Spark-Horovod [IPython notebook](https://github.com/SciDriver/spark-mpi-tf/blob/master/examples/mnist/spark_horovod.ipynb)  for handwritten digit classification (see, for reference, [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/layers)).






