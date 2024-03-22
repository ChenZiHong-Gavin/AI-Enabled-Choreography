from functions.load_data import *
import pyvista as pv
from pyvista import examples

### Generate edges to draw
# Normal, connected skeleton:
skeleton_idxs = []
for g1,g2 in skeleton_lines:
    entry = []
    skeleton_idxs.append([point_labels.index(l) for l in g1] + [point_labels.index(l) for l in g2])

# Cloud of every point connected:
cloud_idxs = []
for i in range(len(point_labels)):
    for j in range(len(point_labels)):
        if i != j:
            cloud_idxs.append([i, j])


# animate a video of the stick figure.
# `ghost` may be a second sequence, which will be superimposed
# on the primary sequence.
# If ghost_shift is given, the primary and ghost sequence will be separated laterally
# by that amount.
# `zcolor` may be an N-length array, where N is the number of vertices in seq, and will
# be used to color the vertices. Typically this is set to the avg. z-value of each vtx.
def animate_stick(seq, ghost=None, ghost_shift=0, edge_types=None, edge_opacities=None, threshold=0, edge_class=None, figsize=None, zcolor=None, pointer=None, ax_lims=(-0.4,0.4), speed=45,
                  dot_size=20, dot_alpha=0.5, lw=2.5, cmap='cool_r', pointer_color='black', cloud=False, cloud_alpha=0.03, skeleton=True, skeleton_alpha=0.3):
    print(seq.shape)
    # seq = seq.reshape(-1, 3)
    # point_cloud = pv.PolyData(seq)
    # point_cloud.plot(eye_dome_lighting=True)
    origin_seq = pv.PolyData(seq[0])
    plotter = pv.Plotter(notebook=False, off_screen=True)

    plotter.open_gif("move.gif")
    for i in range(0, len(seq)):
        plotter.clear()
        plotter.add_mesh(pv.PolyData(seq[i]), color="black", point_size=10, opacity=dot_alpha)
        plotter.write_frame()
    plotter.close()



    # print(points.shape)
    # print(seq.shape)
    # point_cloud = pv.PolyData(points)
    # point_cloud.plot(eye_dome_lighting=True)


if __name__ == "__main__":
    ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data(pattern="../data/mariel_*.npy")
    seq_len = 500
    index_start = 0
    # index_start = np.random.randint(0,len(ds_all_centered)-seq_len)
    print("Seeding with frame {}".format(index_start))
    xtest = ds_all_centered[index_start:index_start + seq_len, :, :3]

    animate_stick(xtest, figsize=(10, 8), cmap='inferno', cloud=False)

    # x = np.arange(-10, 10, 0.5)
    # y = np.arange(-10, 10, 0.5)
    # x, y = np.meshgrid(x, y)
    # r = np.sqrt(x ** 2 + y ** 2)
    # z = np.sin(r)
    #
    # # Create and structured surface
    # grid = pv.StructuredGrid(x, y, z)
    # grid["Height"] = z.ravel()
    #
    #
    # # Create a plotter object and set the scalars to the Z height
    # plotter = pv.Plotter(notebook=False, off_screen=True)
    # plotter.add_mesh(
    #     grid,
    #     scalars="Height",
    #     lighting=False,
    #     show_edges=True,
    #     clim=[-1, 1],
    # )
    #
    # # Open a gif
    # plotter.open_gif("wave.gif")
    #
    # # Update Z and write a frame for each updated position
    # nframe = 15
    # for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    #     z = np.sin(r + phase)
    #     # Update values inplace
    #     grid.points[:, -1] = z.ravel()
    #     grid["Height"] = z.ravel()
    #     # Write a frame. This triggers a render.
    #     plotter.write_frame()
    #
    # # Closes and finalizes movie
    # plotter.close()


