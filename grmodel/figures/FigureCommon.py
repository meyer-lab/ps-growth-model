import seaborn as sns


def getSetup(figsize, gridd):
    from matplotlib import gridspec, pyplot as plt

    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind")

    # Setup plotting space
    f = plt.figure(figsize=figsize)

    # Make grid
    gs1 = gridspec.GridSpec(*gridd)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return (ax, f)


def subplotLabel(ax, letter, hstretch=1):
    ax.text(-0.2 / hstretch, 1.2, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')
