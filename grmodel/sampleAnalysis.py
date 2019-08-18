from .pymcGrowth import GrowthModel
from .pymcInteraction import drugInteractionModel


def readModel(ff, model, interval=True, fit=True, **kwargs):
    """ Load the appropriate model object. """

    if model == "growthModel":
        model = GrowthModel(loadFile=ff)
        model.importData(2, interval=interval)
        if fit:
            model.performFit()
        print("fit: " + ff)
    elif model == "interactionModel":
        model = drugInteractionModel(loadFile=ff, fit=fit, **kwargs)
    else:
        raise ValueError("Wrong model specified")

    return model
