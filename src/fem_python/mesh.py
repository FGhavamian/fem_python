from fem_python import config


class FEMMesh:
    def __init__(self):
        self.num_nodes = self.get_num_nodes()
        self.element_to_nodes_mapping = self.assemble_element_to_node()
        self.node_coordinates = self.assemble_node_coordinates()

    def get_num_nodes(self):
        if config.element_type == "linear":
            num_nodes = config.num_elements + 1
        if config.element_type == "quad":
            num_nodes = 2 * config.num_elements + 1

        return num_nodes

    def assemble_element_to_node(self):
        element_to_node = {}
        for e in range(config.num_elements):
            if config.element_type == "linear":
                element_to_node[e] = list(range(e, e + 2))
            if config.element_type == "quad":
                element_to_node[e] = list(range(2 * e, 2 * e + 3))

        return element_to_node

    def assemble_node_coordinates(self):
        element_length = config.bar_length / config.num_elements

        node_coords = {}
        for n in range(self.num_nodes):
            if config.element_type == "linear":
                distance_between_nodes = element_length
                node_coords[n] = n * distance_between_nodes
            if config.element_type == "quad":
                distance_between_nodes = element_length / 2
                node_coords[n] = n * distance_between_nodes

        return node_coords


if __name__ == "__main__":
    mesh = FEMMesh()

    print(mesh.num_nodes)
    print(mesh.element_to_nodes_mapping)
    print(mesh.node_coordinates)
