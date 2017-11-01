# Methods

All analysis was implemented in Python, and can be found at [https://github.com/meyer-lab/ps-growth-model](https://github.com/meyer-lab/ps-growth-model), release 1.0 (doi: [00.0000/arc0000000](https://doi.org/doi-url)).

## Cell culture

Human lung carcinoma PC9 cells were obtained from Sigma-Aldrich (St. Louis, MO), and H1299 cells were provided from ATCC (Manassas, VA). All cell lines were grown in RMPI-1640 medium supplemented with 10\% fetal bovine serum and 1\% penicillin-streptomycin, at 37$^\circ$C and 5\% CO$_2$.
End-point cell viability was measured using CellTiter Glo reagent according to the manufacturer's instructions (Promega, Madison, WI). Briefly, cells were seeded overnight at a density of 1.5 × 10$^3$ cells/well in 96-well plates, and then treated with the indicated drugs (LC Laboratories, Woburn, MA). After 72 hours, CellTiter Glo reagent was added to each well and luminescence was detected. 
(End-point cell viability assay and apoptosis assay to show the possible lacking information from drug response curves.)

## Time-lapse microscopy

Cells were seeded at 1.5 × 10$^3$ cells/well density in 96-well plates and cultured overnight. The next day, each indicated treatment was added, along with IncuCyte$\textsuperscript{\textregistered}$ Annexin V Green Reagent (Essen BioScience, Ann Arbor, MI) and 300 nM YOYO-3 prepared in media containing 1.25 mM CaCl$_2$ [@Gelles:2016ka, Kim:2010yu]. Cells were then cultured and imaged within the IncuCyte Zoom (Essen BioScience) every three hours. Four field of views were taken per well. Fluorescence images were then thresholded and the fraction of image area with red fluorescence, green fluorescence, or dual fluorescence was quantified using IncuCyte Zoom software (Essen BioScience). Finally, the fraction of area occupied by cells was analyzed by brightfield analysis.

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

## CFSE-based cell proliferation analysis

Cell divisions were measured using the CellTrace$^{TM}$ CFSE Cell Proliferation Kit (Invitrogen, Carlsbad, CA). Cells were labeled with 5 μM carboxyfluorescein diacetate succinimidyl ester (CFSE) according to the manufacturer's protocol. The stained cells were seeded overnight in 60 mm dishes at a density of 2 × 10$^5$ cells per dish, and then treated with indicated drugs next day. After 72 hours, cells were collected and fixed in 4\% paraformaldehyde prior to acquisition on a BD LSRFortessa flow cytometer (BD Biosciences, San Jose, CA). CFSE signal intensity of 1 × 10$^4$ cells was recorded and analyzed to measure cell divisions. Day zero CFSE-labeled cells were prepared to determine initial labeling.

## Combination effect determination
