# Animation: C_MinE as mesh color, M_MinDE as edge color
print("Creating animation...")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import animation
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

from Container import Container

def MakeAnimation(container, time_series, y_mesh, y_border, mesh_label='', border_label=''):

    fig, ax = plt.subplots(figsize=(10, 5))
    triang = Triangulation(container.trimesh.vertices[:, 0], container.trimesh.vertices[:, 1], container.trimesh.simplices)
    t = time_series
    y_mesh = y_mesh  # Mesh concentrations: 
    y_border = y_border  # Border concentrations: 

    # Initial mesh plot
    c_mesh = ax.tripcolor(triang, y_mesh[:, 0], shading='flat', cmap='viridis', vmin=y_mesh.min(), vmax=y_mesh.max())

    # Prepare border segments and initial colors
    border_segments = []
    for start, end in container.trimesh.borders:
        seg = [container.trimesh.vertices[start], container.trimesh.vertices[end]]
        border_segments.append(seg)
    border_colors = y_border[:, 0]

    # Create LineCollection for borders
    c_border = LineCollection(border_segments, array=border_colors, cmap='plasma', linewidths=3)
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
        c_mesh.set_clim(vmin=y_mesh[:, frame].min(), vmax=y_mesh[:, frame].max())
        cb_mesh.update_normal(c_mesh)
        c_border.set_array(y_border[:, frame])
        c_border.set_clim(vmin=y_border[:, frame].min(), vmax=y_border[:, frame].max())
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