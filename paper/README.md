# Paper

The manuscript `main.tex` is a self-contained LaTeX document that includes
figures from the v2 figure suite via:

```latex
\graphicspath{{../outputs/v2/figures/paper/}}
```

so all `\includegraphics{Fn_...}` calls resolve to
`outputs/v2/figures/paper/Fn_...pdf`. Do not move the file; if you do, update
`\graphicspath` accordingly.

## Compile

From this directory (`paper/`):

```bash
pdflatex main.tex
pdflatex main.tex   # second pass for cleveref / hyperref cross-references
```

A bibliography file does not yet exist; references are inline `[CITE: ...]`
placeholders. See the comment block above the references section in
`main.tex`.

## Outstanding `[TODO: ...]` placeholders

Search `main.tex` for `[TODO:`. Each one identifies a number or sub-figure
detail that was not present in a verified artifact at draft time
(typically a per-type breakdown that would require re-reading
`outputs/v2/figures/captions/Fn_*.md` or running the aggregator with an
extra slice). Resolve before submission.
