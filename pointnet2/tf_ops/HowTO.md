
In a conda env with python 3.7:

Install tensorflow 2 with pip:
```
pip install tensorflow-gpu==2.2
```
Install CUDNN with conda:
```
conda install cudnn=7.6
```

Install CUDA 10.1 from [here](https://developer.nvidia.com/cuda-10.1-download-archive-update2). If you have an unusual install path, also set `$CUDA_HOME` environment variable.


Create a symbolic link of tensorflow_framework.so.x to tensorflow_framework.so [(Issue48)](https://github.com/charlesq34/pointnet2/issues/48#issuecomment-608135179): 
```
cd {YOUR_CONDA_PATH}/envs/6dofgrasp_py3/lib/python3.7/site-packages/tensorflow/
ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
```
also export the LD_LIBRARY_PATH here:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{YOUR_CONDA_PATH}/envs/6dofgrasp_py3/lib/python3.7/site-packages/tensorflow
```

### Compile PointNet CUDA kernels
```
cd pointnet2/tf_ops/grouping
sh compile.sh
```
```
cd pointnet2/tf_ops/d_interpolation
sh compile.sh
```
```
cd pointnet2/tf_ops/sampling
sh compile.sh
```

## Tests

Check whether tensorflow_framework.so.x was found in 
```
ldd tf_grouping_so.so
```

Run compiled operation:
```
python tf_grouping_op_test.py
```

If there is no segfault you are good to go