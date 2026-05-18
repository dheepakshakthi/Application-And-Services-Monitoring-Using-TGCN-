import pyvista as pv
import numpy as np

# Shape: (Frames, Nodes, 3) 
node_positions = np.random.rand(100, 20, 3) 
edges = [(0, 1), (1, 2), (2, 3)] # Define your edge connectivity

plotter = pv.Plotter()
plotter.open_movie("tgcn_animation.mp4", framerate=30)

# Initialize lines
lines = pv.PolyData()
lines.points = node_positions[0]
lines.lines = np.hstack([[len(e), e[0], e[1]] for e in edges]).flatten()

actor = plotter.add_mesh(lines, color="blue", line_width=5, render_lines_as_tubes=True)
plotter.show(auto_close=False)

# Animate over time
for frame_idx in range(node_positions.shape[0]):
    lines.points = node_positions[frame_idx]
    plotter.write_frame()
    plotter.render()

plotter.close()
