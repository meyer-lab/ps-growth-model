import seaborn as sns


def getSetup(figsize, gridd):
    from matplotlib import gridspec, pyplot as plt

    sns.set(style="white", font_scale=0.7, color_codes=True, palette="colorblind")

    # Setup plotting space
    f = plt.figure(figsize=figsize)

    # Make grid
    gs1 = gridspec.GridSpec(*gridd)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return (ax, f, gs1)


def subplotLabel(ax, letter, hstretch=1):
    ax.text(-0.2 / hstretch, 1.2, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """ Add cartoon to a figure file. """
    import svgutils.transform as st

    # Overlay Figure 4 cartoon
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)
