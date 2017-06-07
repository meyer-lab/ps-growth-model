import numpy as np

def geweke_single_chain(chain1, chain2=None):
    """
    Perform the Geweke Diagnostic between two univariate chains. If two chains are input 
    instead of one, Student's t-test is performed instead.
    """
    from scipy.stats import ttest_ind

    len0 = chain1.shape[0]
    if chain2 is None:
        chain2 = chain1[int(np.ceil(len0/2)):len0]
        chain1 = chain1[0:int(np.ceil(len0*0.1))]
    _, pvalue = ttest_ind(chain1,chain2)
    return pvalue

