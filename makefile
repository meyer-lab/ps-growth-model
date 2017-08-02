fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates
pan_common = -F pandoc-crossref -F pandoc-citeproc --filter=$(tdir)/figure-filter.py -f markdown ./Manuscript/Text/*.md

THEANO_FLAGS := 'floatX=float32'

pan_common = -F pandoc-crossref -F pandoc-citeproc -f markdown ./Manuscript/Text/*.md
fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates

.PHONY: clean upload test profile testcover all

all: Manuscript/index.html

Manuscript/Manuscript.pdf: Manuscript/Manuscript.tex
	(cd ./Manuscript && latexmk -xelatex -f -quiet)
	rm -f ./Manuscript/Manuscript.b* ./Manuscript/Manuscript.aux ./Manuscript/Manuscript.fls

Manuscript/Manuscript.tex: Manuscript/Text/*.md Manuscript/index.html
	pandoc -s $(pan_common) --template=$(tdir)/default.latex --latex-engine=xelatex -o $@

Manuscript/index.html: Manuscript/Text/*.md
	pandoc -s $(pan_common) -t html5 --mathjax -c ./Templates/kultiad.css --template=$(tdir)/html.template -o $@

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html
	rm -f $(fdir)/Figure*

test:
	nosetests3 -s --with-timer --timer-top-n 5

testcover:
	nosetests3 --with-xunit --with-xcoverage --cover-package=grmodel -s --with-timer --timer-top-n 5

upload:
	echo "Upload stub"

profile:
	python3 -c "from grmodel.pymcGrowth import GrowthModel; grM = GrowthModel(); grM.importData(3); grM.model.profile(grM.model.logpt).summary()"
