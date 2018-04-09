#! /usr/bin/env python3

from grmodel.pymcGrowth import GrowthModel

a = GrowthModel()
a.importData(2)
a.fit()
a.save()
