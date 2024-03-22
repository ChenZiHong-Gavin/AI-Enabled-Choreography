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
# `zcolor` may be an N-length array, where N is the number of vertices in seq, and will
# be used to color the vertices. Typically this is set to the avg. z-value of each vtx.
def animate_stick(seq, ghost=None, ghost_shift=0, edge_types=None, edge_opacities=None, threshold=0, edge_class=None,
                  zcolor=None, pointer=None, ax_lims=(-0.4, 0.4), speed=45,
                  dot_size=20, dot_alpha=0.5, lw=2.5, cmap='cool_r', pointer_color='black', cloud=False,
                  cloud_alpha=0.03, skeleton=True, skeleton_alpha=0.3):
    """
    :param seq:
    :param ghost: a second sequence to superimpose on the primary sequence
    :param ghost_shift: how far to laterally shift the ghost sequence, in dimension x
    :param edge_types:
    :param edge_opacities:
    :param threshold:
    :param edge_class:
    :param zcolor:
    :param pointer:
    :param ax_lims:
    :param speed:
    :param dot_size:
    :param dot_alpha:
    :param lw:
    :param cmap:
    :param pointer_color:
    :param cloud:
    :param cloud_alpha:
    :param skeleton:
    :param skeleton_alpha:
    :return:
    """
    if ghost_shift and ghost is not None:
        ghost[:, :, 0] += ghost_shift
        seq[:, :, 0] -= ghost_shift

    ghost_color = 'blue'

    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.open_gif("move.gif")
    for i in range(0, len(seq)):
        plotter.add_mesh(pv.PolyData(seq[i]), name='point', color="black", point_size=10, opacity=dot_alpha)
        if ghost is not None:
            plotter.add_mesh(pv.PolyData(ghost[i]), name='ghost', color=ghost_color, point_size=10, opacity=dot_alpha)
        plotter.write_frame()
    plotter.close()

    plotter = pv.Plotter()

    def create_frame(value):
        plotter.add_mesh(pv.PolyData(seq[int(value)]), name='point', color="black", point_size=10, opacity=dot_alpha)
        if ghost is not None:
            plotter.add_mesh(pv.PolyData(ghost[int(value)]), name='ghost', color=ghost_color, point_size=10, opacity=dot_alpha)
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
    index_ghost = np.random.randint(0, len(ds_all_centered) - seq_len)
    print("Seeding with frame {}".format(index_ghost))
    xtest_ghost = ds_all_centered[index_ghost:index_ghost + seq_len, :, :3]
    animate_stick(xtest, xtest_ghost, ghost_shift=1,
                  cmap='inferno', cloud=False)
