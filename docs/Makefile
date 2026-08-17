PROJECT := ANLP Doomsday (documents)
TEAM    := ANLPDoomsday
DOCS    := interim proposal mid final

include ../make/common.mk

##@ Build

all: $(DOCS) ## Build all four documents

interim: interim.pdf ## Interim proposal, body 2 pages
proposal: proposal.pdf ## Final proposal, body 3-4 pages
mid: mid.pdf ## Mid report, body 7-8 pages, one optional appendix page
final: final.pdf ## Final report, body at most 8 pages, two optional appendix pages

# Four pdflatex passes: one before bibtex, one to place the bibliography, and two
# to settle cross-references and page numbers after it lands. bibtex is prefixed
# with - so a document with no citations yet still builds.
%.pdf: %.tex refs.bib
	@$(STEP) "$* (pdflatex, bibtex, pdflatex x3)"
	@pdflatex -interaction=nonstopmode $* > /dev/null || { $(FAIL) "$*: see $*.log"; exit 1; }
	@bibtex $* > /dev/null 2>&1 || true
	@pdflatex -interaction=nonstopmode $* > /dev/null
	@pdflatex -interaction=nonstopmode $* > /dev/null
	@pdflatex -interaction=nonstopmode $* > /dev/null
	@$(DONE) "$*.pdf"

##@ Checks

check: ## Check every built document: log, page limit, bibliography
	@./check.sh

check-%: %.pdf ## Check one document, e.g. make check-proposal
	@./check.sh $*

##@ Submission names

# Kept per document on purpose: pdf/$(TEAM)-Interim.pdf is the file submitted on
# 14 August 2026, so it is only replaced deliberately.
submit-interim: interim.pdf ## Copy the build to pdf/$(TEAM)-Interim.pdf
	@mkdir -p pdf && cp interim.pdf pdf/$(TEAM)-Interim.pdf
	@$(DONE) "pdf/$(TEAM)-Interim.pdf"

submit-proposal: proposal.pdf ## Copy the build to pdf/$(TEAM)-Proposal.pdf
	@mkdir -p pdf && cp proposal.pdf pdf/$(TEAM)-Proposal.pdf
	@$(DONE) "pdf/$(TEAM)-Proposal.pdf"

submit-mid: mid.pdf ## Copy the build to pdf/$(TEAM)-Mid.pdf
	@mkdir -p pdf && cp mid.pdf pdf/$(TEAM)-Mid.pdf
	@$(DONE) "pdf/$(TEAM)-Mid.pdf"

submit-final: final.pdf ## Copy the build to pdf/$(TEAM)-Final.pdf
	@mkdir -p pdf && cp final.pdf pdf/$(TEAM)-Final.pdf
	@$(DONE) "pdf/$(TEAM)-Final.pdf"

##@ Housekeeping

clean: ## Remove build artefacts, leaving pdf/ alone
	@rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@rm -f $(addsuffix .pdf,$(DOCS))

.PHONY: all $(DOCS) check submit-interim submit-proposal submit-mid submit-final clean
