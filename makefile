fdir = ./Manuscript/Figures

.PHONY: clean test profile all doc

flist = 1 2 3 4 S1 S2 S3 S4

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

clean:
	rm -rf doc/build/* doc/build/.doc* doc/build/.build* doc/source/grmodel.* doc/source/modules.rst $(fdir)/Figure*

sampleDose:
	python3 -c "from grmodel.pymcDoseResponse import doseResponseModel; M = doseResponseModel(); M.sample()"

test: venv
	. venv/bin/activate && pytest

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=grmodel --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint3 --rcfile=./common/pylintrc grmodel > pylint.log || echo "pylint3 exited with $?")
