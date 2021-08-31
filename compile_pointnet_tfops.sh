CUDA_INCLUDE=' -I/usr/local/cuda/include/'
CUDA_LIB=' -L/usr/local/cuda/lib64/'
TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
cd pointnet2/tf_ops/sampling

nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu \
 ${CUDA_INCLUDE} ${TF_CFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o tf_sampling_so.so tf_sampling.cpp \
 tf_sampling_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB}

echo 'testing sampling'
python3 tf_sampling.py
 
cd ../grouping

nvcc -std=c++11 -c -o tf_grouping_g.cu.o tf_grouping_g.cu \
 ${CUDA_INCLUDE} ${TF_CFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o tf_grouping_so.so tf_grouping.cpp \
 tf_grouping_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB}

echo 'testing grouping'
python3 tf_grouping_op_test.py

 
cd ../3d_interpolation
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -shared -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
echo 'testing interpolate'
python3 tf_interpolate_op_test.py

