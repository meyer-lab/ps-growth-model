"""
This creates Figure 2.
"""

def makeFigure():
    ''' Make Figure 2. '''
    from string import ascii_uppercase
    from .FigureCommon import getSetup, subplotLabel

    # Get list of axis objects
    ax, f, _ = getSetup((7, 6), (5, 6))

    # Blank out for cartoon
    for axis in ax[0:12]:
        axis.axis('off')

    # Show simulation plots (predicted vs experimental)
    simulationPlots(axes=[ax[12], ax[13], ax[14], ax[18], ax[19], ax[20]])

    # Show violin plots for model parameters
    violinPlots(axes=[ax[15], ax[16], ax[17], ax[21], ax[22], ax[23]])

    for ii, item in enumerate([ax[0], ax[12], ax[15], ax[24]]):
        subplotLabel(item, ascii_uppercase[ii])

    # Try and fix overlapping elements
    f.tight_layout(pad=0.1)
    f.show()

    return f


def simulationPlots(axes):
    from ..sampleAnalysis import readModel, simulation
    # Get the list of drugs for 101117_H1299 experiment
    classM, _ = readModel(ff="101117_H1299")
    drugs = list(sorted(set(classM.drugs)))
    drugs.remove('Control')

    # Iterate over each drug
    for i, drug in enumerate(drugs):
        # Make simulation plots and legend
        axis, patches = simulation('101117_H1299', drug, ax=axes[3*i: 3*i+3], unit='nM')
        #Show legend
        axis[2].legend(handles=patches, labelspacing=0.15, prop={'size': 4})


def violinPlots(axes):
    import seaborn as sns
    from ..utils import violinplot
    dfdict, drugs, params = violinplot('101117_H1299')
    # Plot params vs. drug dose
    for j, drug in enumerate(drugs):
        # Get drug
        dfplot = dfdict[drug]
        # Iterate over each parameter in params
        for i, param in enumerate(params):
            idx = 3*j + i
            # Set y-axis confluence limits for each parameter
            if param == 'div':
                axes[idx].set_ylim([-2.5, -1.2])
            elif param == 'deathRate':
                axes[idx].set_ylim([-5.0, -0.7])
            elif param == 'apopfrac':
                axes[idx].set_ylim([0, 1])
            # Make violin plots
            sns.violinplot(x='dose', y=param, data=dfplot, ax=axes[idx], cut=0)
            axes[idx].set_xlabel(drug + ' dose')
            # Rotate dosage labels
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45,
                                      rotation_mode="anchor", ha="right",
                                      position=(0, 0.05), fontsize=4.5)
