NPROCS := 8

pan_common = -F pandoc-crossref -F pandoc-citeproc -f markdown ./Manuscript/Text/*.md
fdir = ./Manuscript/Figures
tdir = ./Manuscript/Templates

.PHONY: clean upload test

clean:
	rm -f ./Manuscript/Manuscript.* ./Manuscript/index.html
	rm -f $(fdir)/Figure*

test:
	python3 -m unittest discover

upload:
	echo "Upload stub"
