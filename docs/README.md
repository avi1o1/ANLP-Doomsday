# Documents

Sources for the written deliverables, ACL style.

| File           | Deliverable      | Due         | Body length                     |
| -------------- | ---------------- | ----------- | ------------------------------- |
| `interim.tex`  | Interim proposal | 14 Aug 2026 | 2 pages, references from page 3 |
| `proposal.tex` | Final proposal   | 28 Aug 2026 | 3-4 pages excluding references  |
| `mid.tex`      | Mid submission   | 2 Oct 2026  | 7-8 pages excluding references  |

`refs.bib` is shared, 50 entries; the 20 added for the mid report are marked with the
source each was checked against. `acl.sty` and `acl_natbib.bst` come from
acl-org/acl-style-files and are bundled, so nothing external is needed beyond a TeX Live
install with `natbib`, `times`, `microtype`, `booktabs`.

Built PDFs under submission names are in `pdf/`.

## Build

```bash
make            # both documents
make interim
make proposal
make submit     # copy builds into pdf/ under submission names
make clean
```

Each target runs pdflatex, bibtex, pdflatex, pdflatex, which is what the bibliography
needs. `acl.sty` already sets `\bibliographystyle`, so the `.tex` files must not.

## Before submitting

Check that the build is clean of errors, warnings, undefined citations and overfull
boxes, that the interim body still ends on page 2, and that the set of cited keys equals
the set of rendered bibliography entries. Every team member uploads separately.

## Implementation notes

The original document notes above are retained. The commands below describe
the current Make targets; the experiment workflow is in the top-level README.

The `.tex` files are proposal/report sources. Their original schedules do not govern
the implementation. The Makefiles build these documents and package submissions;
training runs through `train.sbatch` and the Python CLI.

`refs.bib`, `acl.sty` and `acl_natbib.bst` are bundled. Install TeX Live with `pdflatex`,
`bibtex`, `natbib`, `times`, `microtype` and `booktabs`. Document checks also require
`pdftotext` (Poppler); submission packaging requires `zip`.

From the repository root:

```sh
make help       # List targets; a bare make also displays help
make docs       # Build interim, proposal, mid and final PDFs
make proposal   # Build one document (also: interim, mid, final)
make check      # Check every built document
make clean      # Remove LaTeX build artifacts, retaining docs/pdf/
```

From this directory, use the document targets directly:

```sh
make all
make proposal
make check-proposal
make submit-proposal
```

Each build runs `pdflatex`, `bibtex`, then three more `pdflatex` passes. The checker
checks the log, body page limits and bibliography. `make submit-proposal` copies the
PDF to `pdf/ANLPDoomsday-Proposal.pdf`; `submit-interim`, `submit-mid` and `submit-final`
export the other documents under their submission names.

From the repository root, `make mid-zip` and `make final-zip` create archives under
`dist/`. Export the slides to `slides/ANLPDoomsday-Slides.pdf` before `make final-zip`.
The separately tracked `context/` folder is excluded from these archives.
`make distclean` also removes packaged archives.

Code is organized under `src/retrieval/`, `src/attention/` and `src/routing/`,
with shared modules at the `src/` root. Run code through `csx` or `python -m src`
from the repository root; submit training with `sbatch train.sbatch`.
