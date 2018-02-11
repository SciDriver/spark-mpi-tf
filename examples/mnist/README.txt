
Running Spark-MPI with Jupyter
-------------------------------

export HYDRA_PROXY_PORT=55555

export PYSPARK_DRIVER_PYTHON='jupyter'
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --port=7777'

pyspark --master local[*]

pkill -9 "hydra_pmi_proxy"
