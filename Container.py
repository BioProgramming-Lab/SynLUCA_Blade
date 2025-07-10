# The simulation will be conducted in a 2D manner, with the container represented as a 2D radial slice of the rotational symmetric real container.
# Here we use cylindrical coordinates (x, r, phi) to represent the container, where phi is not used because the rotational symmetry naturally eliminates the need for it.
# The unit of length is in nm, and the unit of time is in seconds.

import numpy as np
from shapely.geometry import Polygon


class Container:

    # init a container with shape parameters stored in a file
    # shape_file: a file containing the shape parameters of the container, one parameter per line, representing the maximum r along the x axis. starting from x=0, x+=1 nm per line.
    # here, the resolution should not be less than 1 nm, which is the base unit of length
    def __init__(self, shape_file: str, resolution: float = 10):

        self.resolution = resolution

        # Read shape parameters from a file
        self.shape = []
        with open(shape_file, 'r') as f:
            for line in f:
                self.shape.append(float(line))

    def establish(self, Voronoi_iterations: int = 100, animation_dir: str = None):
        print("Initializing mesh")
        # Initialize the container by generating border nodes and inner nodes
        self.init_border_nodes()
        self.init_inner_nodes()

        # Relax inner nodes to make them evenly distributed
        self.relax_inner_nodes(Voronoi_iterations=Voronoi_iterations, animation_dir=animation_dir)

        # Triangulate the container
        self.triangulate()
        print("Mesh initialization complete\n", flush=True)

    # generate critical nodes on the border of the container
    def init_border_nodes(self):
        print("\t--Initializing border nodes", flush=True)
        border_length = [self.shape[0]]
        for i in range(len(self.shape) - 1):
            border_length.append(np.sqrt((self.shape[i + 1] - self.shape[i]) **2 + 1))
        border_length.append(self.shape[-1])

        num_border_nodes = int(np.ceil(sum(border_length) / self.resolution))
        border_edge_length = sum(border_length) / num_border_nodes

        self.border_nodes = [[0, 0]]
        self.roof_nodes = []
        current_length = 0
        for i in range(len(border_length)):
            current_length += border_length[i]
            delta_length = current_length - (len(self.border_nodes)-1) * border_edge_length
            while delta_length >= border_edge_length:
                delta_length -= border_edge_length
                if i == 0:
                    x = 0
                    r = self.shape[0] - delta_length
                    self.border_nodes.append([x, r])
                elif i == len(border_length) - 1:
                    x = len(self.shape) - 1
                    r = delta_length
                    self.border_nodes.append([x, r])
                else:
                    x = i - delta_length / border_length[i]
                    r = self.shape[i] - delta_length / border_length[i] * (self.shape[i] - self.shape[i-1])
                    self.border_nodes.append([x, r])
                    self.roof_nodes.append([x, r])

        # handle the error of floating point precision
        if (len(self.shape)-1 - self.border_nodes[-1][0])**2 + self.border_nodes[-1][1]**2 < 1e-6:
            # if the last node is too close to the end of the container, make it exactly at the end
            self.border_nodes[-1] = [len(self.shape) - 1, 0]
        else:
            # if the last node is not at the end of the container, add a node at the end
            self.border_nodes.append([len(self.shape) - 1, 0])

        # add r=0 nodes list
        num_of_r0_nodes = int((np.ceil(len(self.shape)-1) / self.resolution))
        real_r0_resolution = (len(self.shape)-1) / num_of_r0_nodes
        for i in range(1, num_of_r0_nodes):
            x = i * real_r0_resolution
            r = 0
            self.border_nodes.append([x, r])
    
        self.border_nodes = np.array(self.border_nodes)
        self.shape_polygon = Polygon([(x, r) for x, r in self.border_nodes])

    # generate inner nodes of the container using halton sequence
    def init_inner_nodes(self):
        print("\t--Initializing inner nodes", flush=True)
        self.inner_nodes = []
        max_height = [node[1] for node in self.roof_nodes]
        num_inner_nodes = int(np.ceil((sum(self.shape)-(self.shape[0]+self.shape[-1])/2) / self.resolution**2))
        single_section_length = sum(max_height) / (num_inner_nodes + len(max_height))
        for node in self.roof_nodes:
            num_nodes_in_column = int(np.ceil(node[1] / single_section_length))
            section_length_in_column = node[1] / num_nodes_in_column
            for i in range(num_nodes_in_column-1):
                r = section_length_in_column * (i + 1)
                self.inner_nodes.append([node[0], r])
        self.inner_nodes = np.array(self.inner_nodes)

    # make inner nodes evenly distributed in the container
    def relax_inner_nodes(self, Voronoi_iterations: int = 100, animation_dir: str = None):
        print("\t--Relaxing inner nodes", end=' ', flush=True)
        # Relax inner nodes to make them evenly distributed
        from scipy.spatial import Voronoi, voronoi_plot_2d
        import copy

        max_x = max(self.border_nodes[:, 0]) + 10
        max_r = max(self.border_nodes[:, 1]) + 10
        min_x = -10
        min_r = -10
        far_away_nodes = []
        # Generate far enough points outside the container to ensure Voronoi regions are well defined
        for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            x = 10* max_x * np.cos(angle) + max(self.border_nodes[:, 0])/2
            r = 10* max_r * np.sin(angle) + max(self.border_nodes[:, 1])/2
            far_away_nodes.append([x, r])

        data_frames = []
        for i in range(Voronoi_iterations):
            # Create Voronoi diagram using both inner nodes and far away nodes
            vor = Voronoi(np.vstack((self.inner_nodes, far_away_nodes)))

            # clip Voronoi regions to the shape of the container
            vor_polygons = []
            for region in vor.regions:
                if not -1 in region:  # ignore empty regions
                    polygon = Polygon([vor.vertices[i] for i in region])
                    if polygon.intersects(self.shape_polygon):
                        vor_polygons.append(polygon.intersection(self.shape_polygon))

            # update inner nodes to the centroids of the Voronoi regions
            self.inner_nodes = []
            for poly in vor_polygons:
                centroid = poly.centroid
                self.inner_nodes.append([centroid.x, centroid.y])
            self.inner_nodes = np.array(self.inner_nodes)

            # Gather frame for animation
            if animation_dir is not None:
                data_frames.append((copy.deepcopy(vor), copy.deepcopy(vor_polygons)))
            
            if i % (Voronoi_iterations // 10) == 0 or i == Voronoi_iterations - 1:
                print(".", end='', flush=True)

        if animation_dir is not None:
            print("\n\t\t--Saving animation", flush=True, end='')
            from matplotlib import animation
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            def update(frame_data):
                frame_num, (vor, vor_polygons) = frame_data
                ax.clear()
                for poly in vor_polygons:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.5)
                voronoi_plot_2d(vor, ax=ax)
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_r, max_r)
                ax.set_title(f"Voronoi Iteration {frame_num}")
                ax.set_xlabel('x (nm)')
                ax.set_ylabel('r (nm)')
                ax.set_aspect('equal', adjustable='box')
                return ax,

            ani = animation.FuncAnimation(fig, update, frames=enumerate(data_frames), repeat=False, cache_frame_data=False)
            ani.save(f"{animation_dir}/relaxation.mp4", fps=24, writer='ffmpeg')
            plt.close(fig)
        
        print("")

    # Triangulation
    def triangulate(self):
        print("\t--Triangulating the container", flush=True)
        from scipy.spatial import Delaunay
        points = np.vstack((self.border_nodes, self.inner_nodes))
        tri = Delaunay(points[:, :2])

        # get rid of the triangles that are outside the container shape
        valid_simplices = []
        for simplex in tri.simplices:
            if self.shape_polygon.contains(Polygon(tri.points[simplex]).centroid):
                valid_simplices.append(simplex)
        tri.simplices = np.array(valid_simplices)
        self.triangulation = tri

if __name__ == "__main__":
    container = Container('shape.txt', resolution=200)
    container.establish(animation_dir='.')

    print("Number of border nodes:", len(container.border_nodes))
    print("Number of inner nodes:", len(container.inner_nodes))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(container.shape)), np.array(container.shape), label='Container Shape', color='green', linewidth=3, alpha=0.7)
    plt.plot([node[0] for node in container.border_nodes]+[container.border_nodes[0][0]], [node[1] for node in container.border_nodes]+[container.border_nodes[0][1]], color='blue', markersize=5, label='border Nodes')
    plt.scatter([node[0] for node in container.border_nodes], [node[1] for node in container.border_nodes], marker='o', s=50)
    plt.scatter([node[0] for node in container.inner_nodes], [node[1] for node in container.inner_nodes], marker='x', color='red', label='Inner Nodes')

    # plot triangulation
    plt.triplot(container.triangulation.points[:, 0], container.triangulation.points[:, 1], container.triangulation.simplices.copy(), color='orange', alpha=0.5)

    plt.title('Container Shape and border Nodes')
    plt.xlabel('x (nm)')
    plt.ylabel('r (nm)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

            
