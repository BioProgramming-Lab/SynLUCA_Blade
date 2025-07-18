from Container import Container
from Diffusion import DiffusionProperties

import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

# mesh index in y ranges from 0 to num_mesh-1, where num_mesh is the total number of mesh elements
# border index in y ranges from num_mesh to len(y)-1
def MinDE_system_jax(trimesh: Container.TriMesh, t_span, y0):

    num_mesh = len(trimesh.simplices)
    num_border = len(trimesh.borders)
    mesh_diffusion, border_diffusion, to_membrane = DiffusionProperties(trimesh)
    mesh_diffusion = [jnp.array(md) for md in mesh_diffusion]
    border_diffusion = [jnp.array(bd) for bd in border_diffusion]
    to_membrane = jnp.array(to_membrane)

    MinDE_matters = {
        'C_MinD_ADP': (0, num_mesh), 
        'C_MinD_ATP': (num_mesh, 2*num_mesh), 
        'C_MinE': (2*num_mesh, 3*num_mesh), 
        'M_MinD': (3*num_mesh, 3*num_mesh + num_border), 
        'M_MinDE': (3*num_mesh + num_border, 3*num_mesh + 2*num_border)
    }

    k_bind = 0.1 * 1e3
    k_D2D = 0.108/60 * 1e9
    k_E2D = 0.435/60 * 1e9
    k_hydrolysis = 1.925
    k_phosphorylation = 6

    tri_neighbors = jnp.array(trimesh.tri_neighbors_raw)
    adjacent_tri = jnp.array(trimesh.adjacent_tri)

    @jax.jit
    def MinDE_reactions(t, y, args):
        # Unpack state
        C_MinD_ADP = y[MinDE_matters['C_MinD_ADP'][0]:MinDE_matters['C_MinD_ADP'][1]]
        C_MinD_ATP = y[MinDE_matters['C_MinD_ATP'][0]:MinDE_matters['C_MinD_ATP'][1]]
        C_MinE = y[MinDE_matters['C_MinE'][0]:MinDE_matters['C_MinE'][1]]
        M_MinD = y[MinDE_matters['M_MinD'][0]:MinDE_matters['M_MinD'][1]]
        M_MinDE = y[MinDE_matters['M_MinDE'][0]:MinDE_matters['M_MinDE'][1]]

        # Diffusion for cytosolic components
        def cytosol_diff(i):
            neighbors = tri_neighbors[i]
            md = mesh_diffusion[i]
            d_ADP = jnp.sum(md * (C_MinD_ADP[neighbors] - C_MinD_ADP[i]))
            d_ATP = jnp.sum(md * (C_MinD_ATP[neighbors] - C_MinD_ATP[i]))
            d_E   = jnp.sum(md * (C_MinE[neighbors] - C_MinE[i]))
            return d_ADP, d_ATP, d_E
        d_C_MinD_ADP_dt, d_C_MinD_ATP_dt, d_C_MinE_dt = jax.vmap(cytosol_diff, in_axes=(0))(jnp.arange(num_mesh))
        
        # Diffusion for membrane components
        def border_diff(i):
            if i == 0:
                d_M_MinD = border_diffusion[0][0] * (M_MinD[1] - M_MinD[0])
                d_M_MinDE = border_diffusion[0][0] * (M_MinDE[1] - M_MinDE[0])
            elif i == num_border-1:
                d_M_MinD = border_diffusion[-1][0] * (M_MinD[num_border-2] - M_MinD[num_border-1])
                d_M_MinDE = border_diffusion[-1][0] * (M_MinDE[num_border-2] - M_MinDE[num_border-1])
            else:
                d_M_MinD = border_diffusion[i][0] * (M_MinD[i-1] - M_MinD[i]) + border_diffusion[i][1] * (M_MinD[i+1] - M_MinD[i])
                d_M_MinDE = border_diffusion[i][0] * (M_MinDE[i-1] - M_MinDE[i]) + border_diffusion[i][1] * (M_MinDE[i+1] - M_MinDE[i])
            return d_M_MinD, d_M_MinDE
        d_M_MinD_dt, d_M_MinDE_dt = jax.vmap(border_diff)(jnp.arange(num_border))

        # Membrane reactions
        def membrane_react(i):
            tri_idx = adjacent_tri[i]
            # MinD ATP binding to membrane
            delta1 = k_bind * C_MinD_ATP[tri_idx]
            # MinD ATP binding to membrane facilitated by membrane MinD
            delta2 = k_D2D * M_MinD[i] * C_MinD_ATP[tri_idx]
            # MinE binding to membrane mediated by membrane MinD
            delta3 = k_E2D * M_MinD[i] * C_MinE[tri_idx]
            # MinDE ATP hydrolysis
            delta4 = k_hydrolysis * M_MinDE[i]

            # d_M_MinD
            d_M_MinD = delta1 + delta2 - delta3
            # d_M_MinDE
            d_M_MinDE = delta3 - delta4

            # d_C_MinD_ATP
            d_C_MinD_ATP = - (delta1 + delta2) * to_membrane[i]
            # d_C_MinE
            d_C_MinE = - delta3 * to_membrane[i] + delta4 * to_membrane[i]
            # d_C_MinD_ADP
            d_C_MinD_ADP = delta4 * to_membrane[i]

            return d_M_MinD, d_M_MinDE, d_C_MinD_ATP, d_C_MinE, d_C_MinD_ADP

        mem_results = jax.vmap(membrane_react, in_axes=(0))(jnp.arange(len(adjacent_tri)))
        d_M_MinD_dt = d_M_MinD_dt.at[:].add(mem_results[0])
        d_M_MinDE_dt = d_M_MinDE_dt.at[:].add(mem_results[1])
        d_C_MinD_ATP_dt = d_C_MinD_ATP_dt.at[adjacent_tri].add(mem_results[2])
        d_C_MinE_dt = d_C_MinE_dt.at[adjacent_tri].add(mem_results[3])
        d_C_MinD_ADP_dt = d_C_MinD_ADP_dt.at[adjacent_tri].add(mem_results[4])

        # Cytosolic phosphorylation
        delta_phos = k_phosphorylation * C_MinD_ADP
        d_C_MinD_ATP_dt = d_C_MinD_ATP_dt + delta_phos
        d_C_MinD_ADP_dt = d_C_MinD_ADP_dt - delta_phos

        dydt = jnp.concatenate([
            d_C_MinD_ADP_dt,
            d_C_MinD_ATP_dt,
            d_C_MinE_dt,
            d_M_MinD_dt,
            d_M_MinDE_dt
        ])
        return dydt

    # Use diffrax ODE solver
    term = diffrax.ODETerm(MinDE_reactions)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.arange(t_span[0], t_span[1], 0.04))
    progress_bar = diffrax.TqdmProgressMeter()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.1,
        y0=jnp.array(y0),
        saveat=saveat,
        args=None,
        progress_meter=progress_bar
    )
    return sol


if __name__ == "__main__":
    # Example usage
    container = Container('shape.txt', resolution=200)
    container.establish(animation_dir=None)

    key = jrandom.PRNGKey(42)
    num_mesh = len(container.trimesh.simplices)
    num_border = len(container.trimesh.borders)

    key, subkey = jrandom.split(key)
    C_MinD_ADP_y0 = jrandom.uniform(subkey, (num_mesh,)) * 1000

    key, subkey = jrandom.split(key)
    C_MinD_ATP_y0 = jrandom.uniform(subkey, (num_mesh,)) * 1000

    key, subkey = jrandom.split(key)
    C_MinE_y0 = jrandom.uniform(subkey, (num_mesh,)) * 1000

    key, subkey = jrandom.split(key)
    M_MinD_y0 = jrandom.uniform(subkey, (num_border,)) * 1000

    key, subkey = jrandom.split(key)
    M_MinDE_y0 = jrandom.uniform(subkey, (num_border,)) * 1000

    y0 = jnp.concatenate([
        C_MinD_ADP_y0,
        C_MinD_ATP_y0,
        C_MinE_y0,
        M_MinD_y0,
        M_MinDE_y0
    ])

    t_span = (0, 100)  # Time span for the simulation

    sol = MinDE_system_jax(container.trimesh, t_span, y0)
    # Dump the solution to a file, which includes sol.y and sol.t
    with open("MinDE_solution.pkl", "wb") as f:
        import pickle
        pickle.dump({"t": sol.t, "y": sol.y}, f)

    import numpy as np

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