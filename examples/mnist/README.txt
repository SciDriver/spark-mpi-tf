This Spark-MPI example demonstrates the integration of
Horovod's MPI-based deep learning engine with the Spark
platform within the context of the MNIST application.


1. Start the PMI server

pmixsrv -n 2 ./tensorflow_mnist.py &

2. Run pyspark with Jupyter

export PYSPARK_DRIVER_PYTHON='jupyter'
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --port=7777'

pyspark --master local[*]

3. Run spark_horovod.ipynb in a web brower (http://127.0.0.1:7777)

4. Stop the PMI server

pkill -9 "pmixsrv"



