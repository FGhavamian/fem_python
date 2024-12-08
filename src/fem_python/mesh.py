from fem_python import config


class FEMMesh:
    """The 2D mesh of a 1d bar looks like this:
    3---4---5
    | 0 | 1 |
    0---1---2

    the height is 1. And the lengths is 1. Let's say the depth is also 1.
    """

    def __init__(self):
        self.num_nodes = self._get_num_nodes()
        self.num_dofs = self.num_nodes * 2  # 2D model
        self.element_depths = self._get_elements_depths()
        self.element_to_nodes_mapping = self._assemble_element_to_node()
        self.node_coordinates = self._assemble_node_coordinates()
        self.num_integration_points_needed = self._get_num_integration_points_needed()
        self.node_to_dof_mapping = self._make_node_to_dof_mapping()
        self.boundary_nodes_on_the_right = self._get_boundary_nodes_on_the_right()
        self.boundary_nodes_on_the_left = self._get_boundary_nodes_on_the_left()

    def get_node_coords_for_element(self, e):
        element_nodes = self.element_to_nodes_mapping[e]
        node_coords = [self.node_coordinates[node] for node in element_nodes]
        return node_coords

    def _get_num_integration_points_needed(self):
        """Integration points are used to integrate stiffness matrices.
        Stiffness matrix consists of polynomial shap functions. Ideally,
        we would like to have a number of integration points that solve for the
        integral exactly.

        Returns:
            int: Number of integration points
        """
        return 2

    def _get_num_nodes(self):
        return 6
        # if config.element_type == "linear":
        #     num_nodes = config.num_elements + 1
        # if config.element_type == "quad":
        #     num_nodes = 3 + 2 * (config.num_elements - 1)

        # return num_nodes

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

    def _make_node_to_dof_mapping(self):
        node_dof_mapping = {}
        for node in range(self.num_nodes):
            node_dof_mapping[node] = [2 * node, 2 * node + 1]

        return node_dof_mapping

    def _get_elements_depths(self):
        return 1

    def _get_boundary_nodes_on_the_left(self):
        return [0, 3]

    def _get_boundary_nodes_on_the_right(self):
        return [2, 5]


if __name__ == "__main__":
    mesh = FEMMesh()

    print(mesh.num_nodes)
    print(mesh.element_to_nodes_mapping)
    print(mesh.node_coordinates)
    print(mesh.node_to_dof_mapping)
