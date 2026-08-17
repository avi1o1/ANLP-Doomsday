# Predicting When Collection Statistics Help Top-$k$ Selection

Advanced NLP, Monsoon 2026. Team ANLP Doomsday: Aviral Gupta, Mohit Kumar Singh,
Anirudh Sankar, Arihant Tripathy.

Learned sparse retrieval, block-sparse attention over KV pages, and mixture-of-experts
routing all score a query against a set of learned non-negative features by inner product
and keep the top $k$. None of them uses how often a feature is selected across the
collection, which is the statistic BM25 is built on. We build eight concept bases from one
frozen encoder, score each with a function whose IDF, saturation, length-normalisation and
top-$k$ components switch independently, and regress the resulting change in quality on six
cheap measurements of the basis. The fitted predictor is then applied, without refitting, to
KV-page selection and to expert routing, and to Hindi, Bengali, Telugu and Hindi-English
code-mixed retrieval, where the distributions it depends on are known to shift.

## Layout

| Path        | Contents                                                   |
| ----------- | ---------------------------------------------------------- |
| `docs/`     | LaTeX sources for all four written deliverables, ACL style |
| `docs/pdf/` | built PDFs under their submission names                    |
| `slides/`   | viva presentation                                          |
| `Makefile`  | entry point for every command; `make` lists them           |
| `make/`     | shared Makefile pieces, included by the rest               |

## Deliverables

| Deliverable      | Due         | File                        |
| ---------------- | ----------- | --------------------------- |
| Interim proposal | 14 Aug 2026 | `ANLPDoomsday-Interim.pdf`  |
| Final proposal   | 28 Aug 2026 | `ANLPDoomsday-Proposal.pdf` |
| Mid submission   | 30 Sep 2026 | `ANLPDoomsday-Mid.zip`      |
| Final submission | 31 Oct 2026 | `ANLPDoomsday-Final.zip`    |

Each member submits every deliverable separately.

## Build

A bare `make` lists every target with a description, so it is the place to look rather
than this file.

```sh
make            # list the targets
make docs       # build all four documents
make proposal   # build one document
make check      # log, page limits, bibliography consistency
make mid-zip    # package the mid submission
make final-zip  # package the final submission
```

New targets are documented by writing `## description` on the target line, and grouped
with a `##@ Group` line. Set `NO_COLOR=1` for plain output.

Needs TeX Live with `natbib`, `times`, `microtype` and `booktabs`. The ACL style files are
bundled in `docs/`, so nothing external is fetched.

## Models and runs

Links to any uploaded model or logged run go here.
