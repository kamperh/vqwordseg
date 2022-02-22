import matplotlib.pyplot as plt

# The "Tableau 20" colors as RGB:
# http://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html
tableau20 = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), 
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150), 
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148), 
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), 
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
    ]  
  
# Scale the RGB values to the [0, 1] range
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  


def setup_plot():
    """My default setup for publication style plots."""

    fontsize = 12

    plt.rcParams["figure.figsize"]          = 5, 3.5
    plt.rcParams["figure.subplot.bottom"]   = 0.12
    plt.rcParams["figure.subplot.left"]     = 0.12
    plt.rcParams["figure.subplot.right"]    = 0.97
    plt.rcParams["figure.subplot.top"]      = 0.97

    plt.rcParams["font.sans-serif"]         = "Computer Modern Sans serif"
    plt.rcParams["font.serif"]              = "Computer Modern Roman"
    plt.rcParams["font.size"]               = fontsize
    plt.rcParams["text.usetex"]             = True

    plt.rcParams["axes.linewidth"]          = 0.5
    plt.rcParams["grid.color"]              = "grey"
    plt.rcParams["axes.titlesize"]          = "medium"
    # plt.rcParams["legend.fontsize"]         = "medium"
    plt.rcParams["legend.fontsize"]         = "small"
    plt.rcParams["patch.linewidth"]         = 0.5

    plt.rcParams["lines.linewidth"]         = 1.25
    plt.rcParams["lines.markeredgewidth"]   = 0
    plt.rcParams["lines.markersize"]        = 4
    plt.rcParams["text.latex.preamble"]     = r"\usepackage{amsmath}"

    plt.rcParams["savefig.transparent"] = True

    # Foreground
    # for param in ["axes.edgecolor", "text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    #     # plt.rcParams[param] = "#23373b"
    #     plt.rcParams[param] = "#373D3F"

    # Background
    # plt.rcParams["axes.facecolor"]              = "#fafafa"


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)
