# This file calculates the diffusion related properties of each mesh and border edge in the container.
# Because of the rotational symmetry of the container, each border edge actually comprises the lateral surface of a circular frustum or cylinder,
# each mesh is a kind of toroid, and the diffusion properties are thus calculated based on the geometry of these shapes.

from Container import Container
import numpy as np

DiffConst_3D = 10 * 1e6  # Diffusion constant, in nm^2/s
DiffConst_2D = 13 * 1e4  # Diffusion constant, in nm^2/s


neighbor2edge = [[1, 2], [0, 2], [0, 1]]  # vertex indices of the edge connecting the neighboring triangle

# Calculate the surface area of the lateral surface of a section of frustum given by two points
# (using the formular of the area of the lateral surface of a section of frustum,
# but not times \theta as the rotation degree will be cancelled out with volume)
def SurfaceArea(coord1, coord2):
    x1, r1 = coord1
    x2, r2 = coord2
    L = np.sqrt((x2 - x1) ** 2 + (r2 - r1) ** 2)  # slant length of the frustum section
    s = (r1 + r2) * L /2
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
    # Calculate the areas of the interfaces of each triangular frustum
    inter_surface_areas = []
    for receptor in range(len(trimesh.tri_neighbors_raw)):
        inter_surface_areas.append([])
        for donor in range(len(trimesh.tri_neighbors_raw[receptor])):
            if trimesh.tri_neighbors_raw[receptor][donor] != -1:
                area = SurfaceArea(
                    trimesh.vertices[trimesh.simplices[receptor][neighbor2edge[donor][0]]],
                    trimesh.vertices[trimesh.simplices[receptor][neighbor2edge[donor][1]]]
                )
                inter_surface_areas[receptor].append(area)
    
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
    container = Container('shape.txt', resolution=100)
    container.establish(animation_dir=None)

    tri, bor, to_mem = DiffusionProperties(container.trimesh)

    print("simplices:", container.trimesh.simplices[418], "\n")
    # print("borders:", container.trimesh.borders, "\n")
    print("neighbors:", container.trimesh.tri_neighbors[418], "\n")
    # print("adjacent:", container.trimesh.adjacent_tri, "\n")

    print("Mesh Diffusion Properties:", tri[418], "\n")
    # print("Border Diffusion Properties:", bor, "\n")
    # print("To Membrane Properties:", to_mem, "\n")