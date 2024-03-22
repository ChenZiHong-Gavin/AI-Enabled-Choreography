from functions.load_data import *
import pyvista as pv
from pyvista import examples

# Normal, connected skeleton:
skeleton_idxs = []
for g1, g2 in skeleton_lines:
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
def animate_stick(seq, ghost=None, ghost_shift=0, edge_types=None, edge_opacities=None, threshold=0, edge_class=None,
                  figsize=None, zcolor=None, pointer=None, ax_lims=(-0.4, 0.4), speed=45,
                  dot_size=20, dot_alpha=0.5, lw=2.5, cmap='cool_r', pointer_color='black', cloud=False,
                  cloud_alpha=0.03, skeleton=True, skeleton_alpha=0.3):
    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.open_gif("move.gif")
    for i in range(0, len(seq)):
        plotter.clear()
        plotter.add_mesh(pv.PolyData(seq[i]), color="black", point_size=10, opacity=dot_alpha)
        plotter.write_frame()
    plotter.close()

    plotter = pv.Plotter()

    def create_frame(value):
        plotter.add_mesh(pv.PolyData(seq[int(value)]), name='cloud', color="black", point_size=10, opacity=dot_alpha)
        return

    plotter.add_slider_widget(create_frame, [0, len(seq) - 1], title="Frame", value=0, interaction_event="always", style="modern")
    plotter.show()



if __name__ == "__main__":
    ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data(pattern="../data/mariel_*.npy")
    seq_len = 50
    index_start = 0
    index_start = np.random.randint(0, len(ds_all_centered) - seq_len)
    print("Seeding with frame {}".format(index_start))
    xtest = ds_all_centered[index_start:index_start + seq_len, :, :3]
    animate_stick(xtest, figsize=(10, 8), cmap='inferno', cloud=False)
