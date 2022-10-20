import numpy as np
import matplotlib.pyplot as pp
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from scipy.signal import butter, filtfilt



# On choisit le GPU sur lequel le code va tourner, entre 0 et 3
cuda.init()
dev = cuda.Device(1)
contx = dev.make_context()


# Définition du domaine d'étude

x_min = -1
x_max = 1
z_min = -1
z_max = 1
Tf = 10
Nz = 200
Nx = 200

# Mise en forme des pas de discrétisations
if (Nx - 2) % 32 != 0:
    Nx += 32 - (Nx - 2) % 32
if (Nz - 2) % 32 != 0:
    Nz += 32 - (Nz - 2) % 32

# Définition  de l'état initial en cuda
Pn = np.zeros((Nz, Nx))  # Vitesse selon x
Pn = Pn.astype(np.float32)
Pn_gpu = cuda.mem_alloc(Pn.nbytes)
cuda.memcpy_htod(Pn_gpu, Pn)

Pn1 = np.zeros((Nz, Nx))  # Vitesse selon x
Pn1 = Pn1.astype(np.float32)
Pn1_gpu = cuda.mem_alloc(Pn1.nbytes)
cuda.memcpy_htod(Pn1_gpu, Pn1)

Pn1_gpu_cpy = cuda.mem_alloc(Pn1.nbytes)
cuda.memcpy_htod(Pn1_gpu_cpy, Pn1)

# Définition des propriété du sol

v = 1
rho = 1000  # Densité

Rho = rho * np.ones((Nz, Nx))  # Coefficient Mu (i,i+1/2,j,j+1/2)
Rho = Rho.astype(np.float32)
Rho_gpu = cuda.mem_alloc(Rho.nbytes)
cuda.memcpy_htod(Rho_gpu, Rho)

V = v * np.ones((Nz, Nx))  # Coefficient Mu (i,i+1/2,j,j+1/2)
V = V.astype(np.float32)
V_gpu = cuda.mem_alloc(V.nbytes)
cuda.memcpy_htod(V_gpu, V)

# Définition des pas de disrétisation

dz = np.float32((z_max - z_min) / (Nz - 1))
dx = np.float32((x_max - x_min) / (Nx - 1))
dt = np.float32(0.9 * min(dx, dz) / (np.sqrt(2) * v))
Nt = int(Tf / dt) + 1

# Coordonnées de la source et position approximative sur le maillage
x_source = 0
z_source = 0
i_source = np.int32(int((x_source - x_min) / dx))
j_source = np.int32(int((z_source - z_min) / dz))

# Creating dataset
X, Z = np.meshgrid(np.linspace(x_min, x_max, Nx), np.linspace(z_min, z_max, Nz))
x = X[0, :]
x = x.astype(np.float32)
x_gpu = cuda.mem_alloc(x.nbytes)
cuda.memcpy_htod(x_gpu, x)

# Définition des paramétres d'itération en temps
nt = 0
t = np.float32(0)

# Code en CUDA utiliser pour le kernel

mod = SourceModule("""
#include <math.h>

__device__ float sinc(float x)
{
    if (x==0)
    {return(1);}
    else
    {return(sin(x)/x);}
}
__global__ void Calcul_point_miroir(float *coor_miroir, float *z_fantome, float *z_surface,float *x_fantome,float *x_surface)
{
    int idx = (threadIdx.x+1+blockIdx.x*blockDim.x);
    int i=-10;
    float min_dist = (x_fantome[idx]-x_surface[10*idx-10])*(x_fantome[idx]-x_surface[10*idx-10])+
    (z_fantome[idx]-z_surface[10*idx-10])*(z_fantome[idx]-z_surface[10*idx-10]);
    int i_min=-10;
    for (i=-10;i<10;i++) 
        {
        if ((x_fantome[idx]-x_surface[10*idx+i])*(x_fantome[idx]-x_surface[10*idx+i])+
    (z_fantome[idx]-z_surface[10*idx+i])*(z_fantome[idx]-z_surface[10*idx+i])<min_dist)
            {
                min_dist=(x_fantome[idx]-x_surface[10*idx+i])*(x_fantome[idx]-x_surface[10*idx+i])+
    (z_fantome[idx]-z_surface[10*idx+i])*(z_fantome[idx]-z_surface[10*idx+i]);
                i_min=i;
            }
        }
    coor_miroir[2*idx]=2*x_surface[10*idx+i_min]-x_fantome[idx];
    coor_miroir[2*idx+1]=2*z_surface[10*idx+i_min]-z_fantome[idx];
}
__device__ float Df(float xn1,float xn, float dx)
{
    return((xn1-xn)/dx);
}
__device__ float Source(float t,float dt)
{
    float alpha =400.0;
    float t0 = 5.0*t;
    float result=exp(-alpha * (t - t0) * (t-t0));
    return (result);
}
__global__ void iteration_temps(float *pn, float *pn1,float *pn1_cpy, float *rho, float *v, int i_source, int j_source ,float dt ,float dx, 
float dz, float t)
{ 
    int idx = (threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y + 1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2);
    int source = i_source+ j_source*(blockDim.x*gridDim.x+2);
    pn1[idx]=2*pn1_cpy[idx]-pn[idx] + v[idx]*v[idx]*dt*dt*rho[idx]*(Df(Df(pn1_cpy[idx+1],pn1_cpy[idx],dx)/rho[idx+1],
    Df(pn1_cpy[idx],pn1_cpy[idx-1],dx)/rho[idx],dx) + Df(Df(pn1_cpy[idx+(blockDim.x *gridDim.x+2)],pn1_cpy[idx],dz)
    /rho[idx+(blockDim.x *gridDim.x+2)],Df(pn1_cpy[idx],pn1_cpy[idx-(blockDim.x *gridDim.x+2)],dz)/rho[idx],dz));
    if (idx == source)
    {
        float excitation = Source(t,dt);
        pn1[idx] += dt*dt * excitation;
    }
}
__global__ void Mise_a_zero_miroir_velocity (float *p, int *coor_fantome_z)
{
    if(threadIdx.y+blockDim.y *blockIdx.y+1>coor_fantome_z[threadIdx.x+blockDim.x *blockIdx.x+1]-1)
    {
        p[threadIdx.x+1+blockDim.x *blockIdx.x+(threadIdx.y+1+blockDim.y *blockIdx.y)*(blockDim.x *gridDim.x+2)]=0;
    }

}
__global__ void Iteration_miroir_velocity (float *coor_miroir, float *p,float *p_cpy, int *coor_fantome_z,float dx,float dz,int Nz)
{
    int idx=(threadIdx.x+1) +(blockIdx.x*blockDim.x);
    float pvalue=0;
    int i=0;
    int j=0;
    for(i=1;i<blockDim.x *gridDim.x+1;i++)
    {
        for (j=1;j<Nz;j++)
        {
            pvalue+=p_cpy[i+j*(blockDim.x *gridDim.x+2)]*sinc(coor_miroir[2*idx]/(dx)-i)*sinc(coor_miroir[2*idx+1]/(dz)-j);
        }
    }
    p[idx+coor_fantome_z[idx]*(blockDim.x *gridDim.x+2)]=-pvalue;  
}
""")

# Definition des points de la courbe
Nx_courbe = 10 * (Nx - 1) + 1
dx_courbe = np.float32((x_max - x_min) / (Nx_courbe - 1))

x_surface=np.linspace(x_min,x_max,Nx_courbe)
x_surface = x_surface .astype(np.float32)
x_surface_gpu = cuda.mem_alloc(x_surface .nbytes)
cuda.memcpy_htod(x_surface_gpu, x_surface)

z_surface = 0.5*np.sin(5*np.square(x_surface)+0.5)
z_surface = z_surface.astype(np.float32)
z_surface_gpu = cuda.mem_alloc(z_surface.nbytes)
cuda.memcpy_htod(z_surface_gpu, z_surface)

def filtre_surface(data,cutoff,fs,order):
    normal_cutoff=2*cutoff/fs
    b,a=butter(order,normal_cutoff,btype=('low'),analog=False)
    y=filtfilt(b,a,data)
    return(y)

z_surface=filtre_surface(z_surface,1/dx,1/dx_courbe,4)

z_fantome=[]
for i in range (Nx):
    z_fantome.append((int((z_surface[10*i]+z_min)/dz)+1)*dz-z_min)
z_fantome =np.array(z_fantome)
z_fantome = z_fantome.astype(np.float32)
z_fantome_gpu = cuda.mem_alloc(z_fantome.nbytes)
cuda.memcpy_htod(z_fantome_gpu, z_fantome)

Coor_miroir = np.zeros((Nx , 2))
Coor_miroir = Coor_miroir.astype(np.float32)
Coor_miroir_gpu = cuda.mem_alloc(Coor_miroir.nbytes)
cuda.memcpy_htod(Coor_miroir_gpu, Coor_miroir)

Coor_point_fantome = (z_fantome-z_min)/dz
Coor_point_fantome = Coor_point_fantome .astype(np.int32)
Coor_point_fantome_gpu = cuda.mem_alloc(Coor_point_fantome .nbytes)
cuda.memcpy_htod(Coor_point_fantome_gpu, Coor_point_fantome )

# Importation des fonctions cuda
iteration_temps = mod.get_function("iteration_temps")
Calcul_point_miroir = mod.get_function("Calcul_point_miroir")
Mise_a_zero_miroir_velocity=mod.get_function("Mise_a_zero_miroir_velocity")
Iteration_miroir_velocity=mod.get_function("Iteration_miroir_velocity")

longueur_grille_x = (Nx - 2) // 32
longueur_grille_z = (Nz - 2) // 32
hauteur_matrice_z = np.int32(Nz-1)


Calcul_point_miroir(Coor_miroir_gpu, z_fantome_gpu, z_surface_gpu, x_gpu, x_surface_gpu, block=(32, 1, 1), grid=(longueur_grille_x, 1))
# print(Coor_surface)
# cuda.memcpy_dtoh(Coor_fantome_z, Coor_fantome_z_gpu)
# print(Coor_fantome_z)

while nt < Nt:
    # Itération en temps
    nt += 1
    t += dt
    # Calcul sur GPU
    iteration_temps(Pn_gpu, Pn1_gpu, Pn1_gpu_cpy, Rho_gpu, V_gpu, i_source, j_source, dt, dx, dz, t, block=(32, 32, 1),
                    grid=(longueur_grille_x, longueur_grille_z))
    Mise_a_zero_miroir_velocity(Pn1_gpu, Coor_point_fantome_gpu, block=(32, 32, 1), grid=(longueur_grille_x, longueur_grille_z))

    for i in [1, 10]:
        Iteration_miroir_velocity(Coor_miroir_gpu, Pn1_gpu, Pn1_gpu_cpy, Coor_point_fantome_gpu, dx, dz, hauteur_matrice_z, block=(32, 1, 1), grid=(longueur_grille_x, 1))
    cuda.memcpy_dtod(Pn_gpu, Pn1_gpu_cpy, Pn1.nbytes)
    cuda.memcpy_dtod(Pn1_gpu_cpy, Pn1_gpu, Pn1.nbytes)

    # Affichage en python
    if nt % 50 == 0:
        # On importe les résultats Cuda en python
        cuda.memcpy_dtoh(Pn1, Pn1_gpu)

        # Affichage U
        pp.figure(figsize=(8, 6))
        Pmin = np.min(Pn1)
        Pmax = np.max(Pn1)
        mylevelsU = np.linspace(Pmin, Pmax, 30)
        pp.contourf(X, Z, Pn1, levels=mylevelsU, cmap="coolwarm")
        pp.xlabel("X")
        pp.ylabel("Z")
        pp.title("PRESSION")
        pp.show()
contx.pop()
print(1)
