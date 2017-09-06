fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates
pan_common = -s -F pandoc-crossref -F pandoc-citeproc -f markdown

.PHONY: clean test profile testcover all

all: Manuscript/index.html Manuscript/Manuscript.pdf

$(fdir)/Figure%.svg: genFigures.py
	mkdir -p ./Manuscript/Figures
	python3 genFigures.py $*

$(fdir)/Figure%pdf: $(fdir)/Figure%svg
	rsvg-convert -f pdf $< -o $@

Manuscript/Manuscript.pdf: Manuscript/Text/*.md Manuscript/index.html $(fdir)/Figure1.pdf $(fdir)/Figure2.pdf $(fdir)/Figure3.pdf $(fdir)/Figure4.pdf $(fdir)/Figure5.pdf
	(cd ./Manuscript && pandoc $(pan_common) --filter=./Templates/figure-filter.py --template=./Templates/default.latex -Vcsl=./Templates/nature.csl -Vbibliography=./References.bib ./Text/*.md -o Manuscript.pdf)

Manuscript/index.html: Manuscript/Text/*.md $(fdir)/Figure1.svg $(fdir)/Figure2.svg $(fdir)/Figure3.svg $(fdir)/Figure4.svg $(fdir)/Figure5.svg
	pandoc $(pan_common) -t html5 --mathjax -c ./Templates/kultiad.css --template=$(tdir)/html.template -Vcsl=./Manuscript/Templates/nature.csl -Vbibliography=./Manuscript/References.bib ./Manuscript/Text/*.md -o $@

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html $(fdir)/Figure*

test:
	nosetests3 -s --with-timer --timer-top-n 5

testcover:
	nosetests3 --with-xunit --with-xcoverage --cover-package=grmodel -s --with-timer --timer-top-n 5

profile:
	python3 -c "from grmodel.pymcGrowth import GrowthModel; grM = GrowthModel(); grM.importData(3); grM.model.profile(grM.model.logpt).summary()"
