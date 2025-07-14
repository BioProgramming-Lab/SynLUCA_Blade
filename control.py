from Container import Container
from Diffusion import DiffusionProperties

from scipy.integrate import solve_ivp
import numpy as np

# mesh index in y ranges from 0 to num_mesh-1, where num_mesh is the total number of mesh elements
# border index in y ranges from num_mesh to len(y)-1
def MinDE_system(t, y, trimesh: Container.TriMesh, mesh_diffusion, border_diffusion, to_membrane):
    """
    System of ODEs for the diffusion process.
    """
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
    sol = solve_ivp(        MinDE_system,
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
    y0[-1] = 1.0

    t_span = (0, 20)  # Time span for the simulation

    sol = Launch(container.trimesh, t_span, y0)

    def animate_concentration(trimesh: Container.TriMesh, sol):
        # Plotting the results
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        from matplotlib import animation
        from matplotlib.collections import LineCollection
        
        fig, ax = plt.subplots(figsize=(10, 5))
        triang = Triangulation(trimesh.vertices[:, 0], trimesh.vertices[:, 1], trimesh.simplices)
        t = sol.t
        y_mesh = sol.y[:len(trimesh.simplices), :]  # Mesh concentrations
        y_border = sol.y[len(trimesh.simplices):, :]  # Border concentrations

        # Initial plot for mesh
        c_mesh = ax.tripcolor(triang, y_mesh[:, 0], shading='flat', cmap='viridis', vmin=y_mesh.min(), vmax=y_mesh.max())
        cb_mesh = fig.colorbar(c_mesh, ax=ax, label='Mesh Concentration', fraction=0.046, pad=0.04)

        # Plot border as line with independent colorbar
        border_verts = [trimesh.vertices[point] for point in trimesh.borders[0]]
        border_verts.append(trimesh.vertices[trimesh.borders[-1][1]])
        # Prepare border line segments and initial colors
        border_segments = []
        border_colors = []
        for i, (start, end) in enumerate(trimesh.borders):
            seg = [trimesh.vertices[start], trimesh.vertices[end]]
            border_segments.append(seg)
            border_colors.append(y_border[i, 0])

        # Create LineCollection for borders
        c_border = LineCollection(border_segments, array=np.array(border_colors), cmap='plasma', linewidths=3)
        border_coll = ax.add_collection(c_border)
        cb_border = fig.colorbar(c_border, ax=ax, label='Border Concentration', fraction=0.046, pad=0.08)

        ax.set_title(f"t = {t[0]:.2f}")
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('r (nm)')
        ax.set_aspect('equal', adjustable='box')

        def update(frame):
            # Update mesh concentrations
            c_mesh.set_array(y_mesh[:, frame])
            c_mesh.set_clim(vmin=y_mesh[:, frame].min(), vmax=y_mesh[:, frame].max())
            # Update border concentrations
            c_border.set_array(y_border[:, frame])
            c_border.set_clim(vmin=y_border[:, frame].min(), vmax=y_border[:, frame].max())
            ax.set_title(f"t = {t[frame]:.2f}")
            return c_mesh, c_border

        ani = animation.FuncAnimation(fig, update, frames=len(t), interval=40, blit=False)
        plt.show()
        # ani.save("concentration_evolution.mp4", fps=25, writer='ffmpeg')

    # Usage after your simulation:
    if __name__ == "__main__":
        # ... your existing code ...
        animate_concentration(container.trimesh, sol)

