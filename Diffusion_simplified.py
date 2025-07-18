# This file calculates the diffusion related properties of each mesh and border edge in the container.
# Because of the rotational symmetry of the container, each border edge actually comprises the lateral surface of a circular frustum or cylinder,
# each mesh is a kind of toroid, and the diffusion properties are thus calculated based on the geometry of these shapes.

from Container_simplified import Container
import numpy as np

DiffConst_3D = 10 * 1e6  # Diffusion constant, in nm^2/s
DiffConst_2D = 13 * 1e3  # Diffusion constant, in nm^2/s


neighbor2edge = np.array([[1, 2], [0, 2], [0, 1]])  # vertex indices of the edge connecting the neighboring triangle

# Calculate the surface area of the lateral surface of a section of frustum given by two points
# (using the formular of the area of the lateral surface of a section of frustum,
# but not times \theta as the rotation degree will be cancelled out with volume)
def SurfaceArea(coord1: np.ndarray, coord2: np.ndarray):
    diff = coord2 - coord1
    L = np.linalg.norm(diff)
    s = (coord1[1] + coord2[1]) * L / 2
    return s

# Calculate the diffusion factor $Df$, which determines the diffusion flux $I$ as $I=-Df*\Delta C$
# So, here, we calculate the diffusion factor for each interacting mesh and border edge, meaning a factor for each neighbor.
# the diffusion factor of tri-mesh $Df$ = DiffConst * InteractingSurface Area / Volume / centroids distance
# the diffusion factor of border edge $Df$ = DiffConst * InteractingCurve Length / Surface / centroids distance
def DiffusionProperties(trimesh: Container.TriMesh):

    # Calculate the centroids of each triangle and border edge, which will then be used to calculate delta distances
    trimesh.calculate_centroids()
    # Calculate the volumes of each trianglar frustum
    tri_volumes = trimesh.calculate_tri_volumes()

    # Vectorized calculation of interface surface areas between neighboring triangles using numpy.vectorize and SurfaceArea
    tri_edge_vertex_indices = trimesh.simplices[:, neighbor2edge]  # shape (n_tri, 3, 2)
    # Get the coordinates for each edge's two vertices
    edge_coords_1 = trimesh.vertices[tri_edge_vertex_indices[:, :, 0]]  # shape (n_tri, 3, 2)
    edge_coords_2 = trimesh.vertices[tri_edge_vertex_indices[:, :, 1]]  # shape (n_tri, 3, 2)
    # Use numpy.vectorize to apply SurfaceArea to each edge
    surface_area_vec = np.vectorize(SurfaceArea, signature='(n),(n)->()')
    inter_surface_areas = surface_area_vec(edge_coords_1, edge_coords_2)  # shape (n_tri, 3)

    pass # code checked until here

    # Calculate the diffusion factors for each mesh
    mesh_diffusion = []
    for receptor in range(len(trimesh.tri_neighbors)):
        mesh_diffusion.append([])
        for donor in range(len(trimesh.tri_neighbors[receptor])):
            # the diffusion factor $Df$ = DiffConst * InteractingSurface Area / Volume / centroids distance
            mesh_diffusion[receptor].append(
                DiffConst_3D * inter_surface_areas[receptor][donor] / tri_volumes[receptor] /
                np.linalg.norm(trimesh.tri_centroids[receptor] - trimesh.tri_centroids[trimesh.tri_neighbors[receptor][donor]])
            )
    
    # Calculate the diffusion factors for each border edge    
    border_diffusion = []
    border_diffusion.append(
        [DiffConst_2D * trimesh.vertices[trimesh.borders[0][1]][1] / SurfaceArea(trimesh.vertices[trimesh.borders[0][0]], trimesh.vertices[trimesh.borders[0][1]]) / np.linalg.norm(trimesh.bor_centroids[0] - trimesh.bor_centroids[1])])
    for i in range(1, len(trimesh.borders)-1):
        border_diffusion.append([])
        border_diffusion[i].append(DiffConst_2D * trimesh.vertices[trimesh.borders[i][0]][1] / SurfaceArea(trimesh.vertices[trimesh.borders[i][0]], trimesh.vertices[trimesh.borders[i][1]]) / np.linalg.norm(trimesh.bor_centroids[i-1] - trimesh.bor_centroids[i]))
        border_diffusion[i].append(DiffConst_2D * trimesh.vertices[trimesh.borders[i][1]][1] / SurfaceArea(trimesh.vertices[trimesh.borders[i][0]], trimesh.vertices[trimesh.borders[i][1]]) / np.linalg.norm(trimesh.bor_centroids[i] - trimesh.bor_centroids[i+1]))
    border_diffusion.append(
        [DiffConst_2D * trimesh.vertices[trimesh.borders[-1][0]][1] / SurfaceArea(trimesh.vertices[trimesh.borders[-1][0]], trimesh.vertices[trimesh.borders[-1][1]]) / np.linalg.norm(trimesh.bor_centroids[-2] - trimesh.bor_centroids[-1])]
    )

    # Calculate the concentration transfer coefficient for each mesh that is connected to the border
    # This coefficient is calculated as $k$ = Surface Area / Volume
    to_membrane = []
    for membrane_section in range(len(trimesh.adjacent_tri)):
        to_membrane.append(
            SurfaceArea(
                trimesh.vertices[trimesh.borders[membrane_section][0]],
                trimesh.vertices[trimesh.borders[membrane_section][1]]
            ) / tri_volumes[trimesh.adjacent_tri[membrane_section]]
        )
    return mesh_diffusion, border_diffusion, to_membrane

if __name__ == "__main__":
    # Example usage
    container = Container('shape.txt', resolution=2000)
    container.establish(animation_dir=None)

    tri, bor, to_mem = DiffusionProperties(container.trimesh)

    print("simplices:", container.trimesh.simplices[418], "\n")
    # print("borders:", container.trimesh.borders, "\n")
    print("neighbors:", container.trimesh.tri_neighbors[418], "\n")
    # print("adjacent:", container.trimesh.adjacent_tri, "\n")

    print("Mesh Diffusion Properties:", tri[418], "\n")
    # print("Border Diffusion Properties:", bor, "\n")
    # print("To Membrane Properties:", to_mem, "\n")