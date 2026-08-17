# Documents

Sources for the written deliverables, ACL style.

| File           | Deliverable      | Due         | Body length                     |
| -------------- | ---------------- | ----------- | ------------------------------- |
| `interim.tex`  | Interim proposal | 14 Aug 2026 | 2 pages, references from page 3 |
| `proposal.tex` | Final proposal   | 28 Aug 2026 | 3-4 pages excluding references  |

`refs.bib` is shared, 30 entries. `acl.sty` and `acl_natbib.bst` come from
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
