from string import ascii_lowercase
from matplotlib import gridspec, pyplot as plt, rcParams
import seaborn as sns
import svgutils.transform as st


rcParams["xtick.major.pad"] = 1.5
rcParams["ytick.major.pad"] = 1.5
rcParams["xtick.minor.pad"] = 1.5
rcParams["ytick.minor.pad"] = 1.5
rcParams["legend.labelspacing"] = 0.06
rcParams["legend.handlelength"] = 1.0
rcParams["legend.handletextpad"] = 0.6
rcParams["legend.borderaxespad"] = 0.25


def getSetup(figsize, gridd):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

    return (ax, f)


def subplotLabel(axs):
    """ Place subplot labels on figure. """
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """ Add cartoon to a figure file. """
    # Overlay Figure 4 cartoon
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)
