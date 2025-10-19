
# GQA

make TARGET=tk_kernel SRC=gqa_kernel_1024.cpp
python test_python.py 1024 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=gqa_kernel_2048.cpp
python test_python.py 2048 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=gqa_kernel_4096.cpp
python test_python.py 4096 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=gqa_kernel_8192.cpp
python test_python.py 8192 64 8 0 mi355x_gqa_non_causal_fwd.json

make TARGET=tk_kernel SRC=gqa_kernel_16384.cpp
python test_python.py 16384 64 8 0 mi355x_gqa_non_causal_fwd.json

# MHA

make TARGET=tk_kernel SRC=mha_kernel_1024.cpp
python test_python.py 1024 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=mha_kernel_2048.cpp
python test_python.py 2048 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=mha_kernel_4096.cpp
python test_python.py 4096 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=mha_kernel_8192.cpp
python test_python.py 8192 16 16 0 mi355x_mha_non_causal_fwd.json

make TARGET=tk_kernel SRC=mha_kernel_16384.cpp
python test_python.py 16384 16 16 0 mi355x_mha_non_causal_fwd.json


# GQA

make TARGET=tk_kernel SRC=causal_gqa_kernel_1024.cpp
python test_python.py 1024 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=causal_gqa_kernel_2048.cpp
python test_python.py 2048 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=causal_gqa_kernel_4096.cpp
python test_python.py 4096 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=causal_gqa_kernel_8192.cpp
python test_python.py 8192 64 8 1 mi355x_gqa_causal_fwd.json

make TARGET=tk_kernel SRC=causal_gqa_kernel_16384.cpp
python test_python.py 16384 64 8 1 mi355x_gqa_causal_fwd.json

# MHA

make TARGET=tk_kernel SRC=causal_mha_kernel_1024.cpp
python test_python.py 1024 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=causal_mha_kernel_2048.cpp
python test_python.py 2048 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=causal_mha_kernel_4096.cpp
python test_python.py 4096 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=causal_mha_kernel_8192.cpp
python test_python.py 8192 16 16 1 mi355x_mha_causal_fwd.json

make TARGET=tk_kernel SRC=causal_mha_kernel_16384.cpp
python test_python.py 16384 16 16 1 mi355x_mha_causal_fwd.json

