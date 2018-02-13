#!/usr/bin/env python

import sys
import warnings
import matplotlib as plt
import os.path

plt.use('AGG')
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

fdir = './Manuscript/Figures/'

if __name__ == '__main__':
    nameOut = 'Figure' + sys.argv[1]

    exec('from grmodel.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')

    print(fdir + nameOut + '.svg')

    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    print(nameOut + ' is done.')
