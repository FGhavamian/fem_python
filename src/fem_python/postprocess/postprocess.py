import meshio

from fem_python.mesh.mesh import FEMMesh


def write_to_vtk(displacement_vec, force_vec, fem_mesh: FEMMesh):

    results = {"x": [], "y": [], "ux": [], "uy": [], "fx": [], "fy": []}

    for node in range(fem_mesh.num_nodes):
        dof_x = 2 * node
        dof_y = 2 * node + 1

        results["ux"].append(displacement_vec[dof_x])
        results["uy"].append(displacement_vec[dof_y])
        results["fx"].append(force_vec[dof_x])
        results["fy"].append(force_vec[dof_y])

    meshio.write_points_cells(
        "outputs/1d_bar.vtk",
        fem_mesh.node_coords,
        fem_mesh.cells,
        point_data={
            "ux": results["ux"],
            "uy": results["uy"],
            "fx": results["fx"],
            "fy": results["fy"],
        },
    )
