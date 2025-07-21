# Animation: C_MinE as mesh color, M_MinDE as edge color
print("Creating animation...")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import animation
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from tqdm import tqdm

from Container import Container

def MakeAnimation(container, time_series, y_mesh, y_border, mesh_label='', border_label=''):
    fig, ax = plt.subplots(figsize=(10, 5))
    triang = Triangulation(container.trimesh.vertices[:, 0], container.trimesh.vertices[:, 1], container.trimesh.simplices)
    t = time_series

    # LogNorm for color scaling (avoid zero by setting vmin to a small positive value)
    mesh_norm = mcolors.LogNorm(vmin=max(y_mesh.min(), 1e0), vmax=y_mesh.max())
    border_norm = mcolors.LogNorm(vmin=max(y_border.min(), 1e0), vmax=y_border.max())

    # Initial mesh plot with log scale
    c_mesh = ax.tripcolor(triang, y_mesh[:, 0], shading='flat', cmap='viridis', norm=mesh_norm)

    # Prepare border segments and initial colors
    border_segments = []
    for start, end in container.trimesh.borders:
        seg = [container.trimesh.vertices[start], container.trimesh.vertices[end]]
        border_segments.append(seg)
    border_colors = y_border[:, 0]

    # Create LineCollection for borders with log scale
    c_border = LineCollection(border_segments, array=border_colors, cmap='plasma', linewidths=3, norm=border_norm)
    border_coll = ax.add_collection(c_border)

    # Use make_axes_locatable to create colorbars with the same height as the main plot
    divider = make_axes_locatable(ax)
    cax_right = divider.append_axes("right", size="2%", pad=0.05)
    cax_left = divider.append_axes("left", size="2%", pad=1.2)

    cb_mesh = fig.colorbar(c_mesh, cax=cax_right, label=f'{mesh_label} (Mesh)')
    cb_border = fig.colorbar(c_border, cax=cax_left, label=f'{border_label} (Border)')
    cax_left.yaxis.set_label_position('left')
    cax_left.yaxis.tick_left()

    ax.set_title(f"t = {t[0]:.2f}")
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('r (nm)')
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        c_mesh.set_array(y_mesh[:, frame])
        cb_mesh.update_normal(c_mesh)
        c_border.set_array(y_border[:, frame])
        cb_border.update_normal(c_border)
        ax.set_title(f"t = {t[frame]:.2f}")
        return c_mesh, c_border

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=40, blit=False)
    plt.tight_layout()
    # plt.show()

    save_ani = True  # Set to True to save the animation
    if save_ani == True:
        save_dir = "concentration_evolution.mp4"
        print(f'Saving animation to {save_dir}...')
        video_writer = animation.FFMpegWriter(fps=25, codec='h264_videotoolbox', extra_args=['-q:v', '70']) # -q:v for quality
        update_func = lambda _i, _n: progress_bar.update(1)
        progress_bar = tqdm(total=len(time_series), desc='Saving animation', unit='frame')
        ani.save(save_dir, writer=video_writer, dpi=100, progress_callback=update_func)
        progress_bar.close()
        print('Animation saved successfully.')

if __name__ == "__main__":

    import numpy as np

    container = Container('shape.txt', resolution=200)
    container.establish(animation_dir=None)

    t_span = (0, 100)  # Time span for the simulation
    
    # parse the solution from the file
    sol = None
    import pickle
    with open("MinDE_solution.pkl", "rb") as f:
        sol = pickle.load(f)
    print("sol.keys", sol.keys())

    num_mesh = len(container.trimesh.simplices)
    num_border = len(container.trimesh.borders)

    # Parse solution for each component
    C_MinD_ADP = sol["y"][0:num_mesh, :]
    C_MinD_ATP = sol["y"][num_mesh:2*num_mesh, :]
    C_MinE = sol["y"][2*num_mesh:3*num_mesh, :]
    M_MinD = sol["y"][3*num_mesh:3*num_mesh+num_border, :]
    M_MinDE = sol["y"][3*num_mesh+num_border:3*num_mesh+2*num_border, :]


    # To test the conservation of matters, # we can calculate the total amount of each component at each time step
    # total amount of each component = conc_V * volume (for meshes) + conc_A * area (for borders)
    container.trimesh.calculate_centroids()
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
    for i in range(len(sol["t"])):
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
    plt.plot(sol["t"], D_tot, label='D_total')
    plt.plot(sol["t"], C_MinD_ADP_tot, label='C_MinD_ADP_total')
    plt.plot(sol["t"], C_MinD_ATP_tot, label='C_MinD_ATP_total')
    plt.plot(sol["t"], M_MinD_tot, label='M_MinD_total')
    plt.plot(sol["t"], M_MinDE_tot, label='M_MinDE_total')
    plt.xlabel('Time')
    plt.ylabel('Total Amount')
    plt.legend()

    plt.figure()
    plt.plot(sol["t"], E_tot, label='E_total')
    plt.plot(sol["t"], C_MinE_tot, label='C_MinE_total')
    plt.plot(sol["t"], M_MinDE_tot, label='M_MinDE_total')
    plt.xlabel('Time')
    plt.ylabel('Total Amount')
    plt.legend()

    plt.show()

    if_animation = True  # Set to True to create an animation
    if if_animation:
        MakeAnimation(container, sol["t"], C_MinD_ATP, M_MinDE, mesh_label='C_MinD_ATP', border_label='M_MinDE')