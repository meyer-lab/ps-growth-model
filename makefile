fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates
pan_common = -s -F pandoc-crossref -F pandoc-citeproc -f markdown

.PHONY: clean test profile testcover all

all: Manuscript/index.html Manuscript/Manuscript.pdf

$(fdir)/Figure%.svg: genFigures.py grmodel/data/101117_H1299_samples.pkl grmodel/data/initial-data/sampling.pkl
	mkdir -p ./Manuscript/Figures
	python3 genFigures.py $*

$(fdir)/Figure%pdf: $(fdir)/Figure%svg
	rsvg-convert -f pdf $< -o $@

grmodel/data/030317-2_H1299_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/bh8swc75kk0z3b6/030317-2_H1299_samples.pkl?dl=0

grmodel/data/111717_PC9_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/z1xce0kwafa612a/111717_PC9_samples.pkl?dl=0

grmodel/data/101117_H1299_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/zy5tf8lb08j3ojx/101117_H1299_samples.pkl?dl=0

grmodel/data/101117_H1299_ends_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/eiwyq8pi67qut09/101117_H1299_ends_samples.pkl?dl=0

grmodel/data/initial-data/sampling.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/mw0wbt7hekud5b6/sampling.pkl?dl=0

grmodel/data/030317-2-R1_H1299_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/a0al7xal2g6hpcd/030317-2-R1_H1299_samples.pkl?dl=0

grmodel/data/062117_PC9_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/1tdur7ljesn7thg/062117_PC9_samples.pkl?dl=0

grmodel/data/111717_PC9_ends_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/8c1xj33chlhn7tw/111717_PC9_ends_samples.pkl?dl=0

Manuscript/Manuscript.pdf: Manuscript/Text/*.md Manuscript/index.html $(fdir)/Figure1.pdf $(fdir)/Figure2.pdf $(fdir)/Figure3.pdf $(fdir)/Figure4.pdf $(fdir)/Figure5.pdf
	(cd ./Manuscript && pandoc $(pan_common) --filter=./Templates/figure-filter.py --template=./Templates/default.latex -Vcsl=./Templates/nature.csl -Vbibliography=./References.bib ./Text/*.md -o Manuscript.pdf)

Manuscript/index.html: Manuscript/Text/*.md $(fdir)/Figure1.svg $(fdir)/Figure2.svg $(fdir)/Figure3.svg $(fdir)/Figure4.svg $(fdir)/Figure5.svg
	pandoc $(pan_common) -t html5 --mathjax -c ./Templates/kultiad.css --template=$(tdir)/html.template -Vcsl=./Manuscript/Templates/nature.csl -Vbibliography=./Manuscript/References.bib ./Manuscript/Text/*.md -o $@

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html $(fdir)/Figure*
	rm -f grmodel/data/030317-2_H1299_samples.pkl grmodel/data/111717_PC9_samples.pkl grmodel/data/101117_H1299_samples.pkl grmodel/data/101117_H1299_ends_samples.pkl
	rm -f grmodel/data/initial-data/sampling.pkl grmodel/data/030317-2-R1_H1299_samples.pkl grmodel/data/062117_PC9_samples.pkl grmodel/data/111717_PC9_ends_samples.pkl

test:
	nosetests3 -s --with-timer --timer-top-n 5

testcover:
	nosetests3 --with-xunit --with-xcoverage --cover-package=grmodel -s --with-timer --timer-top-n 5

profile:
	python3 -c "from grmodel.pymcGrowth import GrowthModel; grM = GrowthModel(); grM.importData(3); grM.model.profile(grM.model.logpt).summary()"
