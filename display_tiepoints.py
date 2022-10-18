import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def display_tiepoints(input_img1, input_img2, points, confidence=None,
                      show_points=True, title=None):

    def pseudocolor(col_val, min_val, max_val, col_type):
        # Convert value in the range min_val...max_val to a color between red and green.
        fl = float(col_val - min_val) / (max_val - min_val)
        r, g, b = 1 - fl, fl, 0.
        if col_type == "rgb":
            r, g, b = round(r * 255.0), round(g * 255.0), round(b * 255.0)

        return r, g, b

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(input_img1, cmap="gray")
    axarr[1].imshow(input_img2, cmap="gray")

    for idx, row in enumerate(points):
        p1 = (int(row[0]), int(row[1]))
        p2 = (int(row[2]), int(row[3]))

        if confidence is not None:
            val = confidence[idx]
            col_rgba = pseudocolor(val, 0, 1, "rgba")
        else:
            col_rgba = "red"

        if show_points:
            axarr[0].scatter(p1[0], p1[1], color="b")
            axarr[1].scatter(p2[0], p2[1], color="b")

        con = ConnectionPatch(xyA=p1, coordsA=axarr[0].transData, xyB=p2, coordsB=axarr[1].transData, color=col_rgba,
                              alpha=0.1)
        f.add_artist(con)

    if title is not None:
        f.suptitle(title)

    plt.show()
