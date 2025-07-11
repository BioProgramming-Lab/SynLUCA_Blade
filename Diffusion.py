# This file calculates the diffusion related properties of each mesh and border edge in the container.
# Because of the rotational symmetry of the container, each border edge actually comprises the lateral surface of a circular frustum or cylinder,
# each mesh is a kind of toroid, and the diffusion properties are thus calculated based on the geometry of these shapes.

from Container import Container

# Calculate the diffusion properties $D$, which determines the diffusion flux $J$ as $J=-D*\Delta C$
def DiffusionProperties(trimesh: Container.TriMesh):

    # Calculate the centroids of each triangle and border edge, which will then be used to calculate delta distances
    trimesh.calculate_centroids()

    border_diffusion = []
    mesh_diffusion = []

    return border_diffusion, mesh_diffusion