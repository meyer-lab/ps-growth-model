.PHONY: clean test all

flist = 1 2 3 4 S1 S2 S3 S4
flistFull = $(patsubst %, output/Figure%.svg, $(flist))

all: coverage.xml pylint.log output/manuscript.html output/manuscript.pdf

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

output/Figure%.svg: venv genFigures.py
	mkdir -p ./output
	. venv/bin/activate; ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=./manuscript/ --output-directory=./output --log-level=INFO

output/manuscript.html: venv output/manuscript.md $(flistFull)
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--from=markdown --to=html5 --filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
		--bibliography=output/references.json \
		--csl=common/templates/manubot/style.csl \
		--metadata link-citations=true \
		--include-after-body=common/templates/manubot/default.html \
		--include-after-body=common/templates/manubot/plugins/table-scroll.html \
		--include-after-body=common/templates/manubot/plugins/anchors.html \
		--include-after-body=common/templates/manubot/plugins/accordion.html \
		--include-after-body=common/templates/manubot/plugins/tooltips.html \
		--include-after-body=common/templates/manubot/plugins/jump-to-first.html \
		--include-after-body=common/templates/manubot/plugins/link-highlight.html \
		--include-after-body=common/templates/manubot/plugins/table-of-contents.html \
		--include-after-body=common/templates/manubot/plugins/lightbox.html \
		--mathjax \
		--variable math="" \
		--include-after-body=common/templates/manubot/plugins/math.html \
		--include-after-body=common/templates/manubot/plugins/hypothesis.html \
		--output=output/manuscript.html output/manuscript.md

output/manuscript.pdf: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --from=markdown --to=html5 \
	--pdf-engine=weasyprint --pdf-engine-opt=--presentational-hints \
	--filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
	--bibliography=output/references.json \
	--csl=common/templates/manubot/style.csl \
	--metadata link-citations=true \
	--webtex=https://latex.codecogs.com/svg.latex? \
	--include-after-body=common/templates/manubot/default.html \
	--output=output/manuscript.pdf output/manuscript.md

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf doc/build/* doc/build/.doc* doc/build/.build* doc/source/grmodel.* doc/source/modules.rst output
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true

test: venv
	. venv/bin/activate && pytest

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=grmodel --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc grmodel > pylint.log || echo "pylint exited with $?")
