This Spark-MPI example demonstrates the integration of Horovod's MPI-based deep learning engine
with the Spark platform within the context of the MNIST application. 


Running with Jupyter on a single node
-------------------------------------

export HYDRA_PROXY_PORT=55555

export PYSPARK_DRIVER_PYTHON='jupyter'
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --port=7777'

pyspark --master local[*]

