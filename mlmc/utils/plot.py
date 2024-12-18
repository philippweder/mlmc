import matplotlib as mpl
from cycler import cycler

import logging

logger = logging.getLogger(__name__)

# define names for plot styles
NATURE = "nature"
EPFL = "epfl"
SIAM = "siam"
STYLES = [NATURE, EPFL, SIAM]

MARKERS = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
    ".",
    "1",
    "2",
    "3",
    "4",
    "8",
]

NATURE_COLORS = [
    "#E64B35FF",
    "#4DBBD5FF",
    "#00A087FF",
    "#3C5488FF",
    "#F39B7FFF",
    "#8491B4FF",
    "#91D1C2FF",
    "#DC0000FF",
    "#7E6148FF",
    "#B09C85FF",
]

SIAM_COLORS = [
    "#ff0033", #red1
    "#1153a6", #header1
    "#00ff00", #green1
    "#7a008f", #maroon
    "#ff7300", #orange1
    "#194dff", #blue1
    "#110cbe", #blue2
]

NEUTRALS = ["#000000", "#3b3b3b", "#777777", "#b9b9b9", "#ffffff"]

EPFL_COLORS = [
    "#ff0000",
    "#00a79f",
    "#ffae03",
    "#9fd356",
    "#007480",
    "#ff66b3",
    "#201e50",
    "#a80874",
    "#fbc4ab",
    "#909580",
]
EPFL_NEUTRALS = ["#413D3A", "#CAC7C7"]

# slide sizes
FULL_SLIDE_SIZE = (8.5, 4.6)
TWO_THIRD_SLIDE_SIZE = (5.7, 4.6)
HALF_SLIDE_SIZE = (4.74, 4.6)

# report sizes
LINEWIDTH_SIZE = (3.5, 1.5)

# poster sizes
FULL_COL_SIZE = (14.1, 8.41)
THIRD_COL_SIZE = (4.7, 4.7)


def get_colors_and_markers(color_style=NATURE):
    if color_style == NATURE:
        colors = NATURE_COLORS
        neutrals = NEUTRALS

    elif color_style == EPFL:
        colors = EPFL_COLORS
        neutrals = EPFL_NEUTRALS

    elif color_style == SIAM:
        colors = SIAM_COLORS
        neutrals = NEUTRALS

    return colors, neutrals, MARKERS

def change_hex_brightness(color: str, factor: float = 0.5) -> str:
    color = color.lstrip("#")
    rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    new_rgb = [int(min(255, c * factor)) for c in rgb]
    return "#" + "".join([f"{c:02x}" for c in new_rgb])


def set_plot_style(color_style=SIAM, usetex=False):

    if color_style in [NATURE, SIAM]:

        # axes
        if color_style == NATURE:
            mpl.rcParams["axes.prop_cycle"] = cycler(color=NATURE_COLORS)
        elif color_style == SIAM:
            mpl.rcParams["axes.prop_cycle"] = cycler(color=SIAM_COLORS)

        mpl.rcParams["grid.linewidth"] = 0.3
        mpl.rcParams["grid.linestyle"] = "--"

        # legend
        mpl.rcParams["legend.frameon"] = True
        mpl.rcParams["legend.edgecolor"] = "black"
        mpl.rcParams["legend.framealpha"] = 1
        mpl.rcParams["legend.facecolor"] = "white"
        mpl.rcParams["legend.shadow"] = False
        mpl.rcParams["legend.fancybox"] = False
        mpl.rcParams["legend.fontsize"] = 8


        # lines and markers
        mpl.rcParams["lines.linewidth"] = 1.0
        mpl.rcParams["lines.markeredgewidth"] = 0.5
        mpl.rcParams["lines.markeredgecolor"] = "black"
        mpl.rcParams["lines.markersize"] = 4
        mpl.rcParams["scatter.edgecolors"] = "black"

        # ticks
        mpl.rcParams["xtick.labelsize"] = 8
        mpl.rcParams["ytick.labelsize"] = 8
        mpl.rcParams["axes.labelsize"] = 8

        mpl.rcParams["image.cmap"] = "inferno"
        mpl.rcParams["figure.figsize"] = HALF_SLIDE_SIZE

    elif color_style == EPFL:
        # axes
        mpl.rcParams["axes.prop_cycle"] = cycler(color=EPFL_COLORS)
        mpl.rcParams["axes.edgecolor"] = EPFL_NEUTRALS[0]
        mpl.rcParams["axes.titlecolor"] = EPFL_NEUTRALS[0]
        mpl.rcParams["axes.labelcolor"] = EPFL_NEUTRALS[0]

        # lines and markers
        mpl.rcParams["lines.linewidth"] = 1.0
        mpl.rcParams["lines.markeredgecolor"] = EPFL_NEUTRALS[0]
        mpl.rcParams["lines.markeredgewidth"] = 0.5
        mpl.rcParams["lines.markersize"] = 4.0

        # legend
        mpl.rcParams["legend.frameon"] = False
        mpl.rcParams["legend.labelcolor"] = EPFL_NEUTRALS[0]

        # ticks
        mpl.rcParams["xtick.labelcolor"] = EPFL_NEUTRALS[0]
        mpl.rcParams["xtick.color"] = EPFL_NEUTRALS[0]
        mpl.rcParams["ytick.labelcolor"] = EPFL_NEUTRALS[0]
        mpl.rcParams["ytick.color"] = EPFL_NEUTRALS[0]

    if usetex:
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
        mpl.rcParams["text.usetex"] = usetex
        mpl.rcParams["font.family"] = "serif"