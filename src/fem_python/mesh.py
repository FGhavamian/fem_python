import meshio

from fem_python import config


class FEMMesh:
    """The 2D mesh of a 1d bar looks like this:
    3---4---5
    | 0 | 1 |
    0---1---2

    the height is 1. And the lengths is 1. Let's say the depth is also 1.
    """

    def __init__(self):
        mesh = self._load_msh_file()

        self.node_coordinates = mesh.points
        self.num_dofs = len(self.node_coordinates) * config.dofs_per_node

        self.element_to_nodes_mapping = self._get_element_to_nodes_mapping(mesh)

        print(mesh.cell_data_dict["gmsh:physical"])

        print(self.element_to_nodes_mapping)

        # self.boundary_nodes_on_the_right = self._get_boundary_nodes_on_the_right()
        # self.boundary_nodes_on_the_left = self._get_boundary_nodes_on_the_left()

        self.element_depths = self._get_elements_depths()
        self.num_integration_points_needed = self._get_num_integration_points_needed()

    def get_node_coords_for_element(self, e):
        element_nodes = self.element_to_nodes_mapping[e]
        node_coords = [self.node_coordinates[node] for node in element_nodes]
        return node_coords

    def _load_msh_file(self):
        return meshio.read(config.mesh_file_path)

    def _get_num_integration_points_needed(self):
        """Integration points are used to integrate stiffness matrices.
        Stiffness matrix consists of polynomial shap functions. Ideally,
        we would like to have a number of integration points that solve for the
        integral exactly.

        Returns:
            int: Number of integration points
        """
        return 2

    def _get_element_to_nodes_mapping(self, mesh):
        for cell in mesh.cells:
            if cell.dim == 2:
                return cell.data

    def _assemble_element_to_node(self):
        element_to_node = {0: [0, 1, 4, 3], 1: [1, 2, 5, 4]}
        # element_to_node = {}
        # for e in range(config.num_elements):
        #     if config.element_type == "linear":
        #         element_to_node[e] = list(range(e, e + 2))
        #     if config.element_type == "quad":
        #         element_to_node[e] = list(range(2 * e, 2 * e + 3))

        return element_to_node

    def _assemble_node_coordinates(self):
        node_coords = {
            0: (0, 0),
            1: (0.5, 0),
            2: (1, 0),
            3: (0, 1),
            4: (0.5, 1),
            5: (1, 1),
        }
        # node_coords = {}
        # for n in range(self.num_nodes):
        #     if config.element_type == "linear":
        #         distance_between_nodes = self.element_length
        #         node_coords[n] = n * distance_between_nodes
        #     if config.element_type == "quad":
        #         distance_between_nodes = self.element_length / 2
        #         node_coords[n] = n * distance_between_nodes

        return node_coords

    def _get_elements_depths(self):
        return 1

    def _get_boundary_nodes_on_the_left(self):
        return [0, 3]

    def _get_boundary_nodes_on_the_right(self):
        return [2, 5]


if __name__ == "__main__":
    mesh = FEMMesh()
