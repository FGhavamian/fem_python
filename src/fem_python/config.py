num_time_steps = 5
max_num_nr_iterations = 10

material_model_name = "nonlinear_elastic"
bar_elasticity_module = 1
bar_poission_ratio = 0
plane_stress = False

element_type = "Q4"
mesh_file_path = "meshes/1d_bar.msh"

right_boundary_node_tag = 11
left_boundary_node_tag = 12
bottom_boundary_node_tag = 13

prescribed_displacement_at_right_boundary_x = 15

num_integration_points = 2
