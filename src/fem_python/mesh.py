import meshio

from fem_python import config


class FEMMesh:

    def __init__(self):
        mesh = self._load_msh_file()

        self.node_coords = mesh.points[:, :2]

        for cell in mesh.cells:
            if cell.type == "quad":
                self.connectivity_matrix = cell.data

        self.num_nodes = len(self.node_coords)
        self.num_elements = len(self.connectivity_matrix)

    def _load_msh_file(self):
        return meshio.read(config.mesh_file_path)

    def _get_boundary_nodes(self, mesh):
        line_nodes = mesh.cell_data_dict["gmsh:physical"]["line"]

        right_boundary_nodes = []
        left_boundary_nodes = []
        # for node in line_nodes:
        # if node ==


if __name__ == "__main__":
    mesh = FEMMesh()
