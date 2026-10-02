PROJECT := ANLP Doomsday
TEAM    := ANLPDoomsday
DIST    := dist

include make/common.mk

# Everything that goes into a submission zip alongside the write-up. Listed with
# wildcard so the zip targets keep working as directories appear.
CODE = $(wildcard src scripts configs analysis tests results pyproject.toml uv.lock README.md \
	Makefile make .env.example *.sbatch submit-setting-a.sh)

# Nothing generated, cached or private ever enters a zip. context/ is not listed
# in CODE, so it cannot be picked up by accident.
ZIP_EXCLUDE = -x '*/__pycache__/*' '*.pyc' '*/.DS_Store' '*/.pytest_cache/*' \
	'*.egg-info/*' '*/.ruff_cache/*' '*/data/*' '*/results/raw/*'

##@ Documents

docs: ## Build all four documents
	@$(STEP) "building all four documents"
	@$(MAKE) -C docs all
	@$(DONE) "docs/{interim,proposal,mid,final}.pdf"

interim: ## Build the interim proposal, 2 pages, submitted 14 Aug 2026
	@$(MAKE) -C docs interim

proposal: ## Build the final proposal, 3-4 pages, due 28 Aug 2026
	@$(MAKE) -C docs proposal

mid: ## Build the mid report, 7-8 pages, due 2 Oct 2026
	@$(MAKE) -C docs mid

final: ## Build the final report, at most 8 pages, due 31 Oct 2026
	@$(MAKE) -C docs final

check: ## Check every built document: log, page limit, bibliography
	@$(MAKE) -C docs check

# ##@ Experiments
# Targets that run the grid, the two transformer settings and the analysis belong
# here. Document each one with `## description` so `make help` stays the entry
# point, and keep long runs behind an explicit target rather than a dependency.

##@ Submission

mid-zip: docs/pdf/$(TEAM)-Mid.pdf ## Package ANLPDoomsday-Mid.zip: write-up and code
	@$(STEP) "packaging $(TEAM)-Mid.zip"
	@mkdir -p $(DIST)
	@rm -f $(DIST)/$(TEAM)-Mid.zip
	@zip -qj $(DIST)/$(TEAM)-Mid.zip $<
	@zip -qr $(DIST)/$(TEAM)-Mid.zip $(CODE) $(ZIP_EXCLUDE)
	@$(DONE) "$(DIST)/$(TEAM)-Mid.zip"

final-zip: docs/pdf/$(TEAM)-Final.pdf ## Package ANLPDoomsday-Final.zip: write-up, code, slides
	@test -f slides/$(TEAM)-Slides.pdf \
		|| { $(FAIL) "slides/$(TEAM)-Slides.pdf is missing; export the deck there first"; exit 1; }
	@$(STEP) "packaging $(TEAM)-Final.zip"
	@mkdir -p $(DIST)
	@rm -f $(DIST)/$(TEAM)-Final.zip
	@zip -qj $(DIST)/$(TEAM)-Final.zip $< slides/$(TEAM)-Slides.pdf
	@zip -qr $(DIST)/$(TEAM)-Final.zip $(CODE) $(ZIP_EXCLUDE)
	@$(DONE) "$(DIST)/$(TEAM)-Final.zip"

docs/pdf/$(TEAM)-Mid.pdf:
	@$(MAKE) -C docs submit-mid

docs/pdf/$(TEAM)-Final.pdf:
	@$(MAKE) -C docs submit-final

##@ Housekeeping

clean: ## Remove LaTeX build artefacts
	@$(MAKE) -C docs clean
	@$(DONE) "build artefacts removed"

distclean: clean ## Also remove the packaged submission zips
	@rm -rf $(DIST)
	@$(DONE) "$(DIST)/ removed"

# Undocumented alias, kept because `make all` is a reflex.
all: docs

.PHONY: all docs interim proposal mid final check mid-zip final-zip clean distclean
