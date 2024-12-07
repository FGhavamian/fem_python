from fem_python import config


class FEMMesh:
    def __init__(self):
        self.num_nodes = self._get_num_nodes()
        self.element_length = self._get_elements_lengths()
        self.element_to_nodes_mapping = self._assemble_element_to_node()
        self.node_coordinates = self._assemble_node_coordinates()
        self.num_integration_points_needed = self._get_num_integration_points_needed()

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
        if config.element_type == "linear":
            return 1
        if config.element_type == "quad":
            return 2

    def _get_num_nodes(self):
        if config.element_type == "linear":
            num_nodes = config.num_elements + 1
        if config.element_type == "quad":
            num_nodes = 3 + 2 * (config.num_elements - 1)

        return num_nodes

    def _assemble_element_to_node(self):
        element_to_node = {}
        for e in range(config.num_elements):
            if config.element_type == "linear":
                element_to_node[e] = list(range(e, e + 2))
            if config.element_type == "quad":
                element_to_node[e] = list(range(2 * e, 2 * e + 3))

        return element_to_node

    def _assemble_node_coordinates(self):
        node_coords = {}
        for n in range(self.num_nodes):
            if config.element_type == "linear":
                distance_between_nodes = self.element_length
                node_coords[n] = n * distance_between_nodes
            if config.element_type == "quad":
                distance_between_nodes = self.element_length / 2
                node_coords[n] = n * distance_between_nodes

        return node_coords

    def _get_elements_lengths(self):
        return config.bar_length / config.num_elements


if __name__ == "__main__":
    mesh = FEMMesh()

    print(mesh.num_nodes)
    print(mesh.element_to_nodes_mapping)
    print(mesh.node_coordinates)
