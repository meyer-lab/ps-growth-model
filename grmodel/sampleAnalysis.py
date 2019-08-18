import pymc3 as pm
from .pymcGrowth import GrowthModel
from .pymcInteraction import drugInteractionModel


def read_dataset(ff, model, interval=True, **kwargs):
    """ Load the appropriate model object. """

    if model == "growthModel":
        model = GrowthModel(loadFile=ff)
        model.importData(2, interval=interval)
        model.performFit()
        print("fit: " + ff)
    elif model == "interactionModel":
        model = drugInteractionModel(loadFile=ff, **kwargs)
    else:
        raise ValueError("Wrong model specified")

    return model


def readModel(ff=None, model=None, **kwargs):
    """
    Calls read_dataset to load pymc model
    Outputs: (model, table for the sampling results)
    """
    if model is None:
        model = "growthModel"

    model = read_dataset(ff, model, **kwargs)

    try:
        model.samples = model.fit.sample(1000)
    except BaseException:
        print(model)

    df = pm.backends.tracetab.trace_to_dataframe(model.samples)

    return (model, df)
