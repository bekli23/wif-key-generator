import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pycuda.compiler as compiler

kernel_code = """
__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
"""

mod = compiler.SourceModule(kernel_code)
add = mod.get_function("add")

a = np.array([1, 2, 3, 4], dtype=np.int32)
b = np.array([10, 20, 30, 40], dtype=np.int32)
c = np.zeros_like(a)

add(drv.In(a), drv.In(b), drv.Out(c), block=(4, 1, 1))

print("Sum:", c)
