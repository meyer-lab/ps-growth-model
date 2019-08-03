fdir = ./Manuscript/Figures

.PHONY: clean test profile testcover all doc

flist = 1 2 3 4 5 S1 S2 S3 S4 S5

all: $(patsubst %, $(fdir)/Figure%.pdf, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

$(fdir)/Figure%.svg: venv genFigures.py
	mkdir -p ./Manuscript/Figures
	. venv/bin/activate; ./genFigures.py $*

$(fdir)/Figure%pdf: $(fdir)/Figure%svg
	rsvg-convert -f pdf $< -o $@

grmodel/data/030317-2_H1299_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/bh8swc75kk0z3b6/030317-2_H1299_samples.pkl?dl=0

grmodel/data/111717_PC9_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/z1xce0kwafa612a/111717_PC9_samples.pkl?dl=0

grmodel/data/101117_H1299_ends_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/eiwyq8pi67qut09/101117_H1299_ends_samples.pkl?dl=0

grmodel/data/030317-2-R1_H1299_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/a0al7xal2g6hpcd/030317-2-R1_H1299_samples.pkl?dl=0

grmodel/data/062117_PC9_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/1tdur7ljesn7thg/062117_PC9_samples.pkl?dl=0

grmodel/data/111717_PC9_ends_samples.pkl: 
	curl -LSso $@ https://www.dropbox.com/s/8c1xj33chlhn7tw/111717_PC9_ends_samples.pkl?dl=0

clean:
	rm -rf doc/build/* doc/build/.doc* doc/build/.build* doc/source/grmodel.* doc/source/modules.rst $(fdir)/Figure*

dataclean:
	rm -f grmodel/data/*.pkl

sampleDose:
	python3 -c "from grmodel.pymcDoseResponse import doseResponseModel; M = doseResponseModel(); M.sample()"

test: venv
	. venv/bin/activate && pytest

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=grmodel --cov-report xml:coverage.xml

profile:
	python3 -c "from grmodel.pymcGrowth import GrowthModel; grM = GrowthModel(); grM.importData(3); grM.model.profile(grM.model.logpt).summary()"

doc:
	sphinx-apidoc -o doc/source grmodel
	sphinx-build doc/source doc/build
