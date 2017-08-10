#!/usr/bin/env python3

import sys
import warnings
import matplotlib
matplotlib.use('AGG')
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

fdir = './Manuscript/Figures/'


if __name__ == '__main__':
    nameOut = 'Figure' + sys.argv[1]

    exec('from grmodel.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print(nameOut + ' is done.')
