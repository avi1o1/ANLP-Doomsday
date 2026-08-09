# Concept-Space BM25 — proposal sources

Advanced NLP, Monsoon 2026. Two submissions, both in ACL style.

| File | Submission | Deadline | Body length |
|---|---|---|---|
| `interim.tex` | Interim proposal | 14 Aug 2026 | 2 pages |
| `proposal.tex` | Final proposal | 20 Aug 2026 | ~4 pages + references |

Compiled PDFs are in `pdf/`.

## Build

```
make            # builds both
make proposal   # final proposal only
make interim    # interim only
make clean
```

Or by hand, four passes because of BibTeX:

```
pdflatex proposal && bibtex proposal && pdflatex proposal && pdflatex proposal
```

Needs a TeX Live install with `natbib`, `times`, `microtype`, `booktabs`, `amsthm`.
`acl.sty` and `acl_natbib.bst` are bundled, so nothing else to fetch.

## Before you submit

1. Replace the author block in both `.tex` files — currently `Author One … Author Four`
   with a placeholder email pattern.
2. Rename the PDFs to `<TeamNameWithNoSpaces>-Interim.pdf` and
   `<TeamNameWithNoSpaces>-Proposal.pdf`.
3. Every team member uploads separately.
4. Verify the four 2026 citations (`clavie2026latent`, `han2026bm25v`,
   `formal2026splare`, `zong2026tokens`) in `refs.bib`. The whole framing rests on
   them being real and settled; if any has moved venue or been withdrawn, the
   opening paragraph needs rewording.

## Files

- `proposal.tex` — final proposal, trimmed to the page limit
- `proposal_uncut_6page.tex` — earlier 6-page version, kept for reference. Contains
  material cut for length: the representation × scoring grid table, the phased
  timeline table, the reproduction-target table, and a standalone Threats to
  Validity section. Paste back from here if a reviewer wants depth over brevity.
- `interim.tex` — 2-page pitch
- `refs.bib` — shared bibliography, 28 entries
- `acl.sty`, `acl_natbib.bst` — ACL style files (from acl-org/acl-style-files)
