"""
This module handles experimental data for drug interaction.
"""
import pymc3 as pm
import theano.tensor as T
from .pymcGrowth import theanoCore, convSignal, conversionPriors, deathPriors
from .interactionData import readCombo, filterDrugC, dataSplit


def blissInteract(X1, X2, hill, IC50, justAdd=False):
    """ Calculate Bliss additive interaction of two Hill curves. """
    drug_one = T.pow(X1, hill[0]) / (T.pow(IC50[0], hill[0]) + T.pow(X1, hill[0]))
    drug_two = T.pow(X2, hill[1]) / (T.pow(IC50[1], hill[1]) + T.pow(X2, hill[1]))

    if justAdd:
        return drug_one + drug_two

    return drug_one + drug_two - drug_one * drug_two


def build_model(X1, X2, timeV, conv0=0.1, confl=None, apop=None, dna=None):
    """ Builds then returns the PyMC model. """

    assert X1.shape == X2.shape

    M = pm.Model()

    with M:
        conversions = conversionPriors(conv0)
        d, apopfrac = deathPriors(1)

        # hill coefs for drug 1, 2; assumed to be the same for both phenotype
        hill_growth = pm.Lognormal("hill_growth", 0.0, 0.1, shape=2)
        hill_death = pm.Lognormal("hill_death", 0.0, 0.1, shape=2)

        # IL50 for drug 1, 2; assumed to be the same for both phenotype
        IC50_growth = pm.Lognormal("IC50_growth", -1.0, 1.0, shape=2)
        IC50_death = pm.Lognormal("IC50_death", -1.0, 1.0, shape=2)

        # E_con values; first death then growth
        E_con = pm.Lognormal("E_con", -1.0, 1.0, shape=2)

        # Calculate the death rate
        death_rates = E_con[0] * blissInteract(X1, X2, hill_death, IC50_death, justAdd=True)  # pylint: disable=unsubscriptable-object

        # Calculate the growth rate
        growth_rates = E_con[1] * (1 - blissInteract(X1, X2, hill_growth, IC50_growth))  # pylint: disable=unsubscriptable-object

        # Test the dimension of growth_rates
        growth_rates = T.opt.Assert("growth_rates did not match X1 size")(growth_rates, T.eq(growth_rates.size, X1.size))

        lnum, eap, deadapop, deadnec = theanoCore(timeV, growth_rates, death_rates, apopfrac, d)

        # Test the size of lnum
        lnum = T.opt.Assert("lnum did not match X1*timeV size")(lnum, T.eq(lnum.size, X1.size * timeV.size))

        confl_exp, apop_exp, dna_exp = convSignal(lnum, eap, deadapop, deadnec, conversions)

        # Compare to experimental observation
        if confl is not None:
            confl_obs = T.flatten(confl_exp - confl)
            pm.Normal("confl_fit", sd=T.std(confl_obs), observed=confl_obs)
            conflmean = T.mean(confl, axis=1)
            confl_exp_mean = T.mean(confl_exp, axis=1)
            pm.Deterministic("conflResid", (confl_exp_mean - conflmean) / conflmean[0])

        if apop is not None:
            apop_obs = T.flatten(apop_exp - apop)
            pm.Normal("apop_fit", sd=T.std(apop_obs), observed=apop_obs)

        if dna is not None:
            dna_obs = T.flatten(dna_exp - dna)
            pm.Normal("dna_fit", sd=T.std(dna_obs), observed=dna_obs)

        pm.Deterministic("logp", M.logpt)

    return M


class drugInteractionModel:
    """ An interaction model for two drug response. """
    def __init__(self, loadFile="072718_PC9_BYL_PIM", drug1="PIM447", drug2="BYL749", fit=True):

        # Save input data
        self.loadFile = loadFile

        # Load experimental data
        self.df = readCombo(self.loadFile)

        self.df = filterDrugC(self.df, drug1, drug2)

        self.drugs = [drug1, drug2]

        self.X1, self.X2, self.timeV, self.phase, self.red, self.green = dataSplit(self.df)

        if fit:
            # Build pymc model
            self.model = build_model(self.X1, self.X2, self.timeV, 1.0, confl=self.phase, apop=self.green, dna=self.red)

            # Perform pymc fitting given actual data
            self.samples = pm.sampling.sample(tune=1000, chains=2, model=self.model, progressbar=False)
