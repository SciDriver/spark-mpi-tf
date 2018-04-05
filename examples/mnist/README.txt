This Spark-MPI example demonstrates the integration of
Horovod's MPI-based deep learning engine with the Spark
platform within the context of the MNIST application.

******************************************************
Conventional TesnorFlow-based MNIST application
******************************************************

jupyter notebook --no-browser --port=7777
Run tensorflow.ipynb in a web brower (http://127.0.0.1:7777)

******************************************************
Spark-MPI-Horovod application
******************************************************

1. Start the PMI server (with the OpenMPI sparkmpi plugin )

export OMPI_MCA_mca_base_component_path="<OpenMPI installation>/lib/openmpi:<Spark-MPI installation>/lib"

mpirun -n 4 ./mnist_app.py &

2. Run pyspark with Jupyter

export PYSPARK_DRIVER_PYTHON='jupyter'
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --port=7777'

pyspark --master local[*]
Run spark_horovod.ipynb in a web brower (http://127.0.0.1:7777)

3. Stop the PMI server

pkill -9 "mpirun"



