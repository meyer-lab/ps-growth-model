# Methods

All analysis was implemented in Python, and can be found at [https://github.com/meyer-lab/ps-growth-model](https://github.com/meyer-lab/ps-growth-model), release 1.0 (doi: [00.0000/arc0000000](https://doi.org/doi-url)).

## Cell culture

## Time-lapse microscopy

Cells were seeded at XXX density in XXX-well plates and cultured overnight. The next day, each indicated treatment was added, along with XXX μM pSIVA PS reporter and XXX μM YOYO3 [@Gelles:2016ka, Kim:2010yu]. Cells were then cultured and imaged within the IncuCyte Zoom (Essen BioScience) every three hours. One field of view was taken per well. Fluorescence images were then thresholded and the fraction of image area with red fluorescence, green fluorescence, or dual fluorescence was quantified using IncuCyte Zoom software (Essen BioScience). Finally, the fraction of area occupied by cells was analyzed by brightfield analysis.

## Growth model inference


$$ \frac{\delta L}{\delta t} = gL$$

$$ \frac{\delta E}{\delta t} = aL - dE$$

$$ \frac{\delta D_a}{\delta t} = dE$$

$$ \frac{\delta D_n}{\delta t} = bL$$

Where $g = div - deathRate$, $a = deathRate \cdot apopfrac$, and 
$d = deathRate(1-apopfrac)$. 

Integrating these equations provides the following solution:

$$ L(t) = e^{gt}$$

$$ E(t) = c(L-e^{-dt})$$

$$ D_a(t) = \frac{dc(L-1)}{g} + c(e^{-dt}-1)$$

$$ D_n(t) = \frac{d(L-1)}{g}$$

Where $c = \frac{a}{g+d}$. 

[@Salvatier:2016ki]

## Dose-response inference

## Combination effect determination
