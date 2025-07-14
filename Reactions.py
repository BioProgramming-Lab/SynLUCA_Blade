from Container import Container
from Diffusion import DiffusionProperties

from scipy.integrate import solve_ivp
import numpy as np

MinDE_matters = {'C_MinD_ADP': 0, 'C_MinD_ATP': 1, 'C_MinE': 2, 'M_MinD': 3, 'M_MinDE': 4, 'ATP': 5, 'ADP': 6}
k_bind = 1e5  # Rate of MinD binding to the membrane, in nm^2/s
k_phosphorylation = 6  # Rate of MinD phosphorylation, in 1/s

# mesh index in y ranges from 0 to num_mesh-1, where num_mesh is the total number of mesh elements
# border index in y ranges from num_mesh to len(y)-1
def MinDE_system(t, y, trimesh: Container.TriMesh, mesh_diffusion, border_diffusion, to_membrane):

    # Initialize the rate of change
    dydt = np.zeros_like(y)
    num_mesh = len(trimesh.simplices)

    # Calculate the diffusion for each mesh
    for i in range(num_mesh):
        for j in range(len(mesh_diffusion[i])):
            dydt[i] += mesh_diffusion[i][j] * (y[trimesh.tri_neighbors[i][j]] - y[i])

    # Calculate the diffusion for each border edge
    dydt[num_mesh] = border_diffusion[0][0] * (y[num_mesh + 1] - y[num_mesh])
    for i in range(num_mesh+1, len(y)-1):
        dydt[i] += border_diffusion[i-num_mesh][0] * (y[i-1] - y[i])
        dydt[i] += border_diffusion[i-num_mesh][1] * (y[i+1] - y[i])
    dydt[len(y)-1] = border_diffusion[-1][0] * (y[len(y)-2] - y[len(y)-1])

    return dydt

def Launch(trimesh: Container.TriMesh, t_span, y0):
    mesh_diffusion, border_diffusion, to_membrane = DiffusionProperties(trimesh)
    saveat = np.arange(t_span[0], t_span[1], 0.04)
    sol = solve_ivp(    MinDE_system,
        t_span,
        y0,
        args=(trimesh, mesh_diffusion, border_diffusion, to_membrane),
        method='Radau',
        rtol=1e-13,
        atol=1e-20,
        t_eval=saveat
    )
    return sol

if __name__ == "__main__":
    # Example usage
    container = Container('shape.txt', resolution=100)
    container.establish(animation_dir=None)

    # Initial concentrations for each mesh and border
    y0 = np.zeros(len(container.trimesh.simplices) + len(container.trimesh.borders))
    y0[0] = 1.0  # Initial concentration in the first mesh
    y0[400] = 1.0
    y0[-1] = 1.0

    t_span = (0, 100)  # Time span for the simulation

    sol = Launch(container.trimesh, t_span, y0)


