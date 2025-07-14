from Container import Container
from Diffusion import DiffusionProperties

from scipy.integrate import solve_ivp
import numpy as np

# mesh index in y ranges from 0 to num_mesh-1, where num_mesh is the total number of mesh elements
# border index in y ranges from num_mesh to len(y)-1
def Diffusion_system(t, y, trimesh: Container.TriMesh, mesh_diffusion, border_diffusion, to_membrane):
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
    sol = solve_ivp(        Diffusion_system,
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

    def animate_concentration(trimesh: Container.TriMesh, sol):
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        from matplotlib import animation
        from matplotlib.collections import LineCollection
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, ax = plt.subplots(figsize=(10, 5))
        triang = Triangulation(trimesh.vertices[:, 0], trimesh.vertices[:, 1], trimesh.simplices)
        t = sol.t
        y_mesh = sol.y[:len(trimesh.simplices), :]  # Mesh concentrations
        y_border = sol.y[len(trimesh.simplices):, :]  # Border concentrations

        # Initial mesh plot
        c_mesh = ax.tripcolor(triang, y_mesh[:, 0], shading='flat', cmap='viridis', vmin=y_mesh.min(), vmax=y_mesh.max())

        # Prepare border segments and initial colors
        border_segments = []
        for start, end in trimesh.borders:
            seg = [trimesh.vertices[start], trimesh.vertices[end]]
            border_segments.append(seg)
        border_colors = y_border[:, 0]

        # Create LineCollection for borders
        c_border = LineCollection(border_segments, array=border_colors, cmap='plasma', linewidths=3)
        border_coll = ax.add_collection(c_border)

        # Use make_axes_locatable to create colorbars with the same height as the main plot
        divider = make_axes_locatable(ax)
        cax_right = divider.append_axes("right", size="2%", pad=0.05)
        cax_left = divider.append_axes("left", size="2%", pad=1.2)

        cb_mesh = fig.colorbar(c_mesh, cax=cax_right, label='Mesh Concentration')
        cb_border = fig.colorbar(c_border, cax=cax_left, label='Border Concentration')
        cax_left.yaxis.set_label_position('left')
        cax_left.yaxis.tick_left()

        ax.set_title(f"t = {t[0]:.2f}")
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('r (nm)')
        ax.set_aspect('equal', adjustable='box')

        def update(frame):
            c_mesh.set_array(y_mesh[:, frame])
            c_mesh.set_clim(vmin=y_mesh[:, frame].min(), vmax=y_mesh[:, frame].max())
            cb_mesh.update_normal(c_mesh)
            c_border.set_array(y_border[:, frame])
            c_border.set_clim(vmin=y_border[:, frame].min(), vmax=y_border[:, frame].max())
            cb_border.update_normal(c_border)
            ax.set_title(f"t = {t[frame]:.2f}")
            return c_mesh, c_border

        ani = animation.FuncAnimation(fig, update, frames=len(t), interval=40, blit=False)
        plt.tight_layout()
        plt.show()
        ani.save("concentration_evolution.mp4", fps=25, writer='ffmpeg')

    # Usage after your simulation:
    if __name__ == "__main__":
        # ... your existing code ...
        animate_concentration(container.trimesh, sol)

