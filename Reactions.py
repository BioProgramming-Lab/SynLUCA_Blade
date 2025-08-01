from Container import Container
from Diffusion import DiffusionProperties
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# mesh index in y ranges from 0 to num_mesh-1, where num_mesh is the total number of mesh elements
# border index in y ranges from num_mesh to len(y)-1
def MinDE_system(trimesh: Container.TriMesh, t_span, y0):

    num_mesh = len(trimesh.simplices)
    num_border = len(trimesh.borders)
    mesh_diffusion, border_diffusion, to_membrane = DiffusionProperties(trimesh)

    MinDE_matters = {
        'C_MinD_ADP': (0, num_mesh), 
        'C_MinD_ATP': (num_mesh, 2*num_mesh), 
        'C_MinE': (2*num_mesh, 3*num_mesh), 
        'M_MinD': (3*num_mesh, 3*num_mesh + num_border), 
        'M_MinDE': (3*num_mesh + num_border, 3*num_mesh + 2*num_border)
        }

    k_bind = 0.1 * 1e3  # Rate of cytosolic MinD_ATP binding to the membrane, in nm/s
    k_D2D = 0.108/60 * 1e9 # Rate of cytosolic MinD_ATP bind to the membrane that facilitated by membrane bound MinD, in nm^3/s
    k_E2D = 0.435/60 * 1e9 # Rate of cytosolic MinE bind to the membrane that mediated by membrane bound MinD, in nm^3/s
    k_hydrolysis = 1.925 # Rate of MinDE ATP hydrolysis, in 1/s
    k_phosphorylation = 6  # Rate of MinD phosphorylation, in 1/s

    #last_time = [t_span[0]]
    #pbar = tqdm(total=t_span[1], desc="Integrating MinDE system", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    last_time = [t_span[0]]
    pbar = tqdm(total=t_span[1], desc="Integrating MinDE system", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    def MinDE_reactions(t, y):

        # Unpack state
        C_MinD_ADP = y[MinDE_matters['C_MinD_ADP'][0]:MinDE_matters['C_MinD_ADP'][1]]
        C_MinD_ATP = y[MinDE_matters['C_MinD_ATP'][0]:MinDE_matters['C_MinD_ATP'][1]]
        C_MinE = y[MinDE_matters['C_MinE'][0]:MinDE_matters['C_MinE'][1]]
        M_MinD = y[MinDE_matters['M_MinD'][0]:MinDE_matters['M_MinD'][1]]
        M_MinDE = y[MinDE_matters['M_MinDE'][0]:MinDE_matters['M_MinDE'][1]]

        d_C_MinD_ADP_dt = np.zeros_like(C_MinD_ADP)
        d_C_MinD_ATP_dt = np.zeros_like(C_MinD_ATP)
        d_C_MinE_dt = np.zeros_like(C_MinE)
        d_M_MinD_dt = np.zeros_like(M_MinD)
        d_M_MinDE_dt = np.zeros_like(M_MinDE)
        # Vectorized diffusion for cytosolic components
        d_C_MinD_ADP_dt = np.sum(np.where(trimesh.tri_neighbors_mask, mesh_diffusion, 0) * (C_MinD_ADP[trimesh.tri_neighbors] - C_MinD_ADP[:, np.newaxis]), axis=-1)
        d_C_MinD_ATP_dt = np.sum(np.where(trimesh.tri_neighbors_mask, mesh_diffusion, 0) * (C_MinD_ATP[trimesh.tri_neighbors] - C_MinD_ATP[:, np.newaxis]), axis=-1)
        d_C_MinE_dt = np.sum(np.where(trimesh.tri_neighbors_mask, mesh_diffusion, 0) * (C_MinE[trimesh.tri_neighbors] - C_MinE[:, np.newaxis]), axis=-1)

        # Calculate the diffusion term for each membrane component
        d_M_MinD_dt = np.sum(np.where(trimesh.bor_neighbors_mask, border_diffusion, 0) * (M_MinD[trimesh.bor_neighbors] - M_MinD[:, np.newaxis]), axis=-1)
        d_M_MinDE_dt = np.sum(np.where(trimesh.bor_neighbors_mask, border_diffusion, 0) * (M_MinDE[trimesh.bor_neighbors] - M_MinDE[:, np.newaxis]), axis=-1)

        # Calculate the reaction terms that happen on the membrane
        ## MinD_ATP binding to the membrane
        ## MinD_ATP binding to the membrane is also facilitated by membrane bound MinD
        delta_MinD_ATP_to_membrane = (k_bind  +  k_D2D * M_MinD) * C_MinD_ATP[trimesh.adjacent_tri]
        d_M_MinD_dt += delta_MinD_ATP_to_membrane
        d_C_MinD_ATP_dt[trimesh.adjacent_tri] -= delta_MinD_ATP_to_membrane * to_membrane

        ## MinE binding to the membrane mediated by membrane bound MinD
        delta_MinE_to_membrane = k_E2D * M_MinD * C_MinE[trimesh.adjacent_tri]
        d_M_MinDE_dt += delta_MinE_to_membrane
        d_M_MinD_dt -= delta_MinE_to_membrane
        d_C_MinE_dt[trimesh.adjacent_tri] -= delta_MinE_to_membrane * to_membrane

        ## MinDE ATP hydrolysis
        delta_MinDE_hydrolysis = k_hydrolysis * M_MinDE
        d_M_MinDE_dt -= delta_MinDE_hydrolysis
        d_C_MinD_ADP_dt[trimesh.adjacent_tri] += delta_MinDE_hydrolysis * to_membrane
        d_C_MinE_dt[trimesh.adjacent_tri] += delta_MinDE_hydrolysis * to_membrane

        # Calculate the reaction terms that happen in the cytosol
        ## MinD ATP phosphorylation
        delta_C_MinD_ADP_to_ATP = k_phosphorylation * C_MinD_ADP
        d_C_MinD_ADP_dt -= delta_C_MinD_ADP_to_ATP
        d_C_MinD_ATP_dt += delta_C_MinD_ADP_to_ATP
        
        # Assemble the rate of change for each component
        dydt = np.concatenate([
            d_C_MinD_ADP_dt,
            d_C_MinD_ATP_dt,
            d_C_MinE_dt,
            d_M_MinD_dt,
            d_M_MinDE_dt
        ])
        
        if t - last_time[0] >= 0.04:  # Update progress bar every 0.04 seconds
            pbar.update(t - last_time[0])
            last_time[0] = t
        return dydt
    
    # Use scipy's solve_ivp to solve the ODE system
    saveat = np.arange(t_span[0], t_span[1], 0.04)
    sol = solve_ivp(
        MinDE_reactions,
        t_span=t_span,
        y0=y0,
        method='BDF',
        rtol=1e-3,
        atol=0,
        t_eval=saveat
    )
    pbar.close()
    return sol

if __name__ == "__main__":
    # Example usage
    container = Container('shape.txt', resolution=100)
    container.establish(animation_dir=None)

    np.random.seed(42)  # For reproducibility
    # Initial concentrations for each mesh and border
    C_MinD_ADP_y0 = np.random.rand(len(container.trimesh.simplices)) * 1000
    C_MinD_ATP_y0 = np.random.rand(len(container.trimesh.simplices)) * 1000
    C_MinE_y0 = np.random.rand(len(container.trimesh.simplices)) * 1000
    M_MinD_y0 = np.random.rand(len(container.trimesh.borders)) * 1000
    M_MinDE_y0 = np.random.rand(len(container.trimesh.borders)) * 1000
    y0 = np.concatenate([
        C_MinD_ADP_y0,
        C_MinD_ATP_y0,
        C_MinE_y0,
        M_MinD_y0,
        M_MinDE_y0
    ])

    t_span = (0, 100)  # Time span for the simulation

    sol = MinDE_system(container.trimesh, t_span, y0)
    # Dump the solution to a file, which includes sol.y and sol.t
    with open("MinDE_solution.pkl", "wb") as f:
        import pickle
        pickle.dump({"t": sol.t, "y": sol.y}, f)

    num_mesh = len(container.trimesh.simplices)
    num_border = len(container.trimesh.borders)

    # Parse solution for each component
    C_MinD_ADP = sol.y[0:num_mesh, :]
    C_MinD_ATP = sol.y[num_mesh:2*num_mesh, :]
    C_MinE = sol.y[2*num_mesh:3*num_mesh, :]
    M_MinD = sol.y[3*num_mesh:3*num_mesh+num_border, :]
    M_MinDE = sol.y[3*num_mesh+num_border:3*num_mesh+2*num_border, :]


    # To test the conservation of matters, # we can calculate the total amount of each component at each time step
    # total amount of each component = conc_V * volume (for meshes) + conc_A * area (for borders)
    tri_volumes = container.trimesh.calculate_tri_volumes()

    # directly copied from Diffusion.py
    def SurfaceArea(coord1, coord2):
        x1, r1 = coord1
        x2, r2 = coord2
        L = np.sqrt((x2 - x1) ** 2 + (r2 - r1) ** 2)  # slant length of the frustum section
        s = (r1 + r2) * L /2
        return s

    border_areas = []
    for membrane_section in range(len(container.trimesh.adjacent_tri)):
        border_areas.append(
            SurfaceArea(
                container.trimesh.vertices[container.trimesh.borders[membrane_section][0]],
                container.trimesh.vertices[container.trimesh.borders[membrane_section][1]]
            )
        )
    
    E_tot = []
    D_tot = []
    C_MinD_ATP_tot = []
    C_MinD_ADP_tot = []
    C_MinE_tot = []
    M_MinD_tot = []
    M_MinDE_tot = []
    # Calculate the total amount of each component at each time step
    for i in range(len(sol.t)):
        E_tot.append(
            np.sum(C_MinE[:, i] * tri_volumes) + np.sum(M_MinDE[:, i] * border_areas)
        )
        D_tot.append(
            np.sum(C_MinD_ADP[:, i] * tri_volumes) + np.sum(C_MinD_ATP[:, i] * tri_volumes) +
            np.sum(M_MinD[:, i] * border_areas) + np.sum(M_MinDE[:, i] * border_areas)
        )
        C_MinD_ATP_tot.append(np.sum(C_MinD_ATP[:, i] * tri_volumes))
        C_MinD_ADP_tot.append(np.sum(C_MinD_ADP[:, i] * tri_volumes))
        C_MinE_tot.append(np.sum(C_MinE[:, i] * tri_volumes))
        M_MinD_tot.append(np.sum(M_MinD[:, i] * border_areas))
        M_MinDE_tot.append(np.sum(M_MinDE[:, i] * border_areas))

    # Plot the total amount of each component over time
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(sol.t, D_tot, label='D_total')
    plt.plot(sol.t, C_MinD_ADP_tot, label='C_MinD_ADP_total')
    plt.plot(sol.t, C_MinD_ATP_tot, label='C_MinD_ATP_total')
    plt.plot(sol.t, M_MinD_tot, label='M_MinD_total')
    plt.plot(sol.t, M_MinDE_tot, label='M_MinDE_total')
    plt.xlabel('Time')
    plt.ylabel('Total Amount')
    plt.legend()

    plt.figure()
    plt.plot(sol.t, E_tot, label='E_total')
    plt.plot(sol.t, C_MinE_tot, label='C_MinE_total')
    plt.plot(sol.t, M_MinDE_tot, label='M_MinDE_total')
    plt.xlabel('Time')
    plt.ylabel('Total Amount')
    plt.legend()

    # plt.show()

    if_animation = True  # Set to True to create an animation
    if if_animation:
        from Animation import MakeAnimation
        MakeAnimation(container, sol.t, C_MinE, M_MinDE)