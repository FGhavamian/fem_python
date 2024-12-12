import meshio
import numpy as np

from fem_python import config


class FEMMesh:

    def __init__(self):
        mesh = self._load_msh_file()

        self.node_coords = mesh.points[:, :2]

        self.connectivity_matrix = mesh.cells_dict["quad"]

        self.boundary_nodes = self._get_boundary_nodes(mesh)

        self.num_nodes = len(self.node_coords)
        self.num_elements = len(self.connectivity_matrix)

    def _load_msh_file(self):
        return meshio.read(config.mesh_file_path)

    def _get_boundary_nodes(self, mesh):
        lines_physical_tags = mesh.cell_data_dict["gmsh:physical"]["line"]

        line_connectivity = mesh.cells_dict["line"]
        boundary_nodes = {"left": [], "right": []}

        for n in range(len(line_connectivity)):
            if lines_physical_tags[n] == 12:
                boundary_nodes["left"].append(line_connectivity[n])
            elif lines_physical_tags[n] == 11:
                boundary_nodes["right"].append(line_connectivity[n])

        boundary_nodes["left"] = np.array(boundary_nodes["left"])
        boundary_nodes["right"] = np.array(boundary_nodes["right"])

        return boundary_nodes


if __name__ == "__main__":
    mesh = FEMMesh()
    print(mesh.boundary_nodes)
