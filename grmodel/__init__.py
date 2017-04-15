__version__ = '0.0.1'

try:
    import numpy
except ImportError as ie:
    print(ie, '\nNumPy does not seem to be installed.')

try:
    import scipy
except ImportError as ie:
    print(ie, '\nSciPy does not seem to be installed.')

try:
    import pandas
except ImportError as ie:
    print(ie, '\nPandas does not seem to be installed.')

try:
    import memoize
except ImportError as ie:
    print(ie, '\nmemoize2 does not seem to be installed.')

try:
    import matplotlib
except ImportError as ie:
    print(ie, '\nmatplotlib does not seem to be installed.')
