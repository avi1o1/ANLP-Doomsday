# Slides

The final viva needs a presentation, preferably as a PDF inside the submission zip.
Slides for the mid viva are optional.

Export the deck as `ANLPDoomsday-Slides.pdf` in this directory; `make final-zip` at
the repository root picks it up from there.

## Implementation notes

Slide export is manual. Training uses `sbatch train.sbatch`; it does not build
slides or documents. The code lives in `src/retrieval/`, `src/attention/` and
`src/routing/`, with shared modules at the `src/` root.

See [the implementation guide](../context/IMPLEMENTATION.md) for experiments
and saved artifacts, and [the document guide](../docs/README.md) for packaging.
