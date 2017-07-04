NPROCS := 8

THEANO_FLAGS := 'floatX=float32'

pan_common = -F pandoc-crossref -F pandoc-citeproc -f markdown ./Manuscript/Text/*.md
fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates

.PHONY: clean upload test profile testcover

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html
	rm -f $(fdir)/Figure*

test:
	nosetests -s --with-timer --timer-top-n 5

testcover:
	nosetests --with-xunit --with-xcoverage --cover-package=grmodel -s --with-timer --timer-top-n 5

upload:
	echo "Upload stub"

profile:
	python3 -c "from grmodel.pymcGrowth import GrowthModel; grM = GrowthModel(); grM.importData(3); grM.model.profile(grM.model.logpt).summary()"