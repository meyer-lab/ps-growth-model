.PHONY: clean all

flist = 4 3 S3 1 2 S1 S2 S4
flistFull = $(patsubst %, output/Figure%.svg, $(flist))
pandocCommon = -f markdown \
	--bibliography=output/references.json \
	--csl=style.csl -F pandoc-fignos -F pandoc-eqnos -F pandoc-tablenos \
	--metadata link-citations=true

all: pylint.log output/manuscript.pdf output/manuscript.docx output/manuscript.md

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	@test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Uqr requirements.txt
	@touch venv/bin/activate

output/Figure%.svg: venv genFigures.py grmodel/figures/Figure%.py
	@mkdir -p ./output
	. venv/bin/activate; ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=./manuscript/ --output-directory=./output --log-level=INFO

output/manuscript.pdf: venv output/manuscript.md $(flistFull) style.csl
	. venv/bin/activate && pandoc -t html5 $(pandocCommon) \
	--pdf-engine=weasyprint --pdf-engine-opt=--presentational-hints \
	--webtex=https://latex.codecogs.com/svg.latex? \
	--include-after-body=common/templates/manubot/default.html \
	-o $@ output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistFull) style.csl
	. venv/bin/activate && pandoc --verbose -t docx $(pandocCommon) \
		--reference-doc=common/templates/manubot/default.docx \
		--resource-path=.:content \
		-o $@ output/manuscript.md

style.csl: 
	curl -so $@ -L https://www.zotero.org/styles/nature?source=1

clean:
	@mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf output venv style.csl
	@mkdir output
	@mv requests-cache.sqlite output/requests-cache.sqlite || true

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc grmodel > pylint.log || echo "pylint exited with $?")
