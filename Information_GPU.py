import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

taille_x = np.int(0)
taille_y = np.int(0)
taille_z = np.int(0)
taille_x_gpu = cuda.mem_alloc(8)
taille_y_gpu = cuda.mem_alloc(8)
taille_z_gpu = cuda.mem_alloc(8)

mod = SourceModule("""
    __global__ void taille(unsigned int *taille_x,unsigned int *taille_y,unsigned int *taille_z)
    {   taille_x->blockDim.x;
        taille_y=blockDim.y;
        taille_z=blockDim.z;
    }
""")
func = mod.get_function("taille")
func(taille_x_gpu, taille_y_gpu, taille_z_gpu)
cuda.memcpy_dtoh(taille_x, taille_x_gpu)
cuda.memcpy_dtoh(taille_y, taille_y_gpu)
cuda.memcpy_dtoh(taille_z, taille_z_gpu)
print([taille_x,taille_y, taille_z])
