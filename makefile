NPROCS := 8

pan_common = -F pandoc-crossref -F pandoc-citeproc -f markdown ./Manuscript/Text/*.md
fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates

.PHONY: clean upload test profile testcover

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html
	rm -f $(fdir)/Figure*
	rm -f profile.png profile.pstats

test:
	nosetests -s --with-timer --timer-top-n 5

testcover:
	nosetests --with-xunit --with-xcoverage --cover-package=grmodel -s --with-timer --timer-top-n 5

upload:
	echo "Upload stub"

profile:
	python3 -m cProfile -o profile.pstats Benchmark.py
	python3 -m gprof2dot -n 2.0 -f pstats profile.pstats | dot -Tpng -o profile.png