import gmsh


gmsh.initialize()

gmsh.model.add("1d_bar")

# These four points are corner nodes the 1D bar
# Note that lc determines the density of mesh around the point.
lc = 1e-2
p_bl = gmsh.model.geo.addPoint(0, 0, 0, lc)
p_br = gmsh.model.geo.addPoint(1, 0, 0, lc)
p_tr = gmsh.model.geo.addPoint(1, 1, 0, lc)
p_tl = gmsh.model.geo.addPoint(0, 1, 0, lc)

l_b = gmsh.model.geo.addLine(p_bl, p_br)
l_r = gmsh.model.geo.addLine(p_br, p_tr)
l_t = gmsh.model.geo.addLine(p_tr, p_tl)
l_l = gmsh.model.geo.addLine(p_tl, p_bl)

cl = gmsh.model.geo.addCurveLoop([l_b, l_r, l_t, l_l])
s = gmsh.model.geo.addPlaneSurface([cl])

num_elements_x = 10
num_elements_y = 3
gmsh.model.geo.mesh.setTransfiniteCurve(l_t, num_elements_x)
gmsh.model.geo.mesh.setTransfiniteCurve(l_b, num_elements_x)
gmsh.model.geo.mesh.setTransfiniteCurve(l_l, num_elements_y)
gmsh.model.geo.mesh.setTransfiniteCurve(l_r, num_elements_y)


gmsh.model.geo.mesh.setTransfiniteSurface(s)

gmsh.model.geo.mesh.setRecombine(2, s)

gmsh.model.geo.synchronize()

# Note that meshio (the library that we use to read this mesh), doesn't read name argument below.
# For this reason, we have to assign tags to the physical groups.
# We use these tags to identify physical groups when reading the mesh.
gmsh.model.addPhysicalGroup(1, [l_r], tag=11, name="right_boundary")
gmsh.model.addPhysicalGroup(1, [l_l], tag=12, name="left_boundary")
gmsh.model.addPhysicalGroup(2, [s], tag=1, name="suraface")

gmsh.model.mesh.generate(2)

gmsh.write("meshes/1d_bar.msh")

gmsh.fltk.run()

gmsh.finalize()
