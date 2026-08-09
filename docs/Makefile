ALL = proposal interim

all: $(ALL)

proposal: proposal.pdf
interim: interim.pdf

%.pdf: %.tex refs.bib
	pdflatex -interaction=nonstopmode $*
	-bibtex $*
	pdflatex -interaction=nonstopmode $*
	pdflatex -interaction=nonstopmode $*

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.pdf

.PHONY: all proposal interim clean
