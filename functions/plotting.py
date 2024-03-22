import numpy as np

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

all_idxs = skeleton_idxs + cloud_idxs


# calculate the coordinates for the lines
def get_line_segments(seq):
    xline = np.zeros((seq.shape[0], len(all_idxs), 3, 2))
    for edge,(joint1,joint2) in enumerate(all_idxs):
        xline[:, edge, :, 0] = seq[:, joint1]
        xline[:, edge, :, 1] = seq[:, joint2]
    return xline


# animate a video of the stick figure.
def animate_stick(seq, ghost=None, ghost_shift=0,
                  zcolor=None, speed=45,
                  dot_size=7, dot_alpha=0.5, lw=2.5, skeleton=True):
    """
    :param seq: a sequence of 3D points to animate
    :param ghost: a second sequence to superimpose on the primary sequence
    :param ghost_shift: how far to laterally shift the ghost sequence, in dimension x
    :param zcolor: an N-length array, where N is the number of vertices in seq, and will be used to color the vertices.
    :param speed:
    :param dot_size: size of the dots
    :param dot_alpha: transparency of the dots
    :param lw: line width
    :param skeleton: whether to draw the skeleton
    :return:
    """
    if zcolor is None:
        zcolor = np.zeros(seq.shape[1])

    if ghost_shift and ghost is not None:
        ghost[:, :, 0] += ghost_shift
        seq[:, :, 0] -= ghost_shift

    x_lines = get_line_segments(seq)
    ghost_lines = get_line_segments(ghost) if ghost is not None else None

    def draw(pl, seq, ghost, i):
        pl.add_points(seq, name='point', color="black", point_size=dot_size, opacity=dot_alpha, render_points_as_spheres=True)
        if ghost is not None:
            pl.add_points(ghost, name='ghost', color='blue', point_size=dot_size, opacity=dot_alpha, render_points_as_spheres=True)
        if skeleton:
            line_points = np.zeros((len(skeleton_idxs) * 2, 3))
            for j in range(len(skeleton_idxs)):
                x = np.linspace(x_lines[i, j, 0, 0], x_lines[i, j, 0, 1], 2)
                y = np.linspace(x_lines[i, j, 1, 0], x_lines[i, j, 1, 1], 2)
                z = np.linspace(x_lines[i, j, 2, 0], x_lines[i, j, 2, 1], 2)
                points = np.column_stack((x, y, z))
                line_points[j * 2:(j + 1) * 2] = points
            plotter.add_lines(line_points, name='skeleton', color='purple', width=lw)
            if ghost is not None:
                line_points = np.zeros((len(skeleton_idxs) * 2, 3))
                for j in range(len(skeleton_idxs)):
                    x = np.linspace(ghost_lines[i, j, 0, 0], ghost_lines[i, j, 0, 1], 2)
                    y = np.linspace(ghost_lines[i, j, 1, 0], ghost_lines[i, j, 1, 1], 2)
                    z = np.linspace(ghost_lines[i, j, 2, 0], ghost_lines[i, j, 2, 1], 2)
                    points = np.column_stack((x, y, z))
                    line_points[j * 2:(j + 1) * 2] = points
                plotter.add_lines(line_points, name='ghost_skeleton', color='red', width=lw)



    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.open_gif("move.gif")
    for i in range(0, len(seq)):
        draw(plotter, seq[i], ghost[i] if ghost is not None else None, i)
        plotter.write_frame()


    plotter.close()

    plotter = pv.Plotter()

    def create_frame(value):
        # plotter.add_points(pv.PolyData(seq[int(value)]), name='point', color="black", point_size=dot_size, opacity=dot_alpha, render_points_as_spheres=True)
        # if ghost is not None:
        #     plotter.add_points(pv.PolyData(ghost[int(value)]), name='ghost', color=ghost_color, point_size=dot_size, opacity=dot_alpha, render_points_as_spheres=True)

        draw(plotter, seq[int(value)], ghost[int(value)] if ghost is not None else None, int(value))
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
    animate_stick(xtest, xtest_ghost, ghost_shift=1)
