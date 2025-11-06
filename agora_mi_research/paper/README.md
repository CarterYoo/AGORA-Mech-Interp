# Paper: Mechanistic Interpretability Analysis of RAG Systems

## LaTeX Compilation

### Requirements

```bash
# Install LaTeX distribution
# Windows: MikTeX or TeX Live
# Linux: texlive-full
# Mac: MacTeX

# Or use Overleaf (recommended)
```

### Compile Locally

```bash
cd paper

# Full compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk
latexmk -pdf main.tex
```

### Upload to Overleaf

1. Create new project on Overleaf
2. Upload all files from `paper/` directory
3. Set compiler to pdfLaTeX
4. Compile

## Files

- `main.tex`: Main paper content
- `references.bib`: BibTeX references
- `appendix.tex`: Appendices and supplementary material
- `acl.sty`: ACL style file (download from ACL)
- `acl_natbib.bst`: ACL bibliography style

## ACL Template

Download official ACL template from:
https://github.com/acl-org/acl-style-files

Required files:
- `acl.sty`
- `acl_natbib.bst`

Place in `paper/` directory.

## Structure

### Main Paper

1. Abstract
2. Introduction
3. Related Work
4. Methodology
5. Experiments
6. Results
7. Discussion
8. Conclusion
9. Ethical Considerations
10. Acknowledgments

### Appendices

A. Experimental Details
B. Additional Results
C. Implementation Details
D. Ablation Studies
E. Error Analysis
F. Annotation Examples

## Figures and Tables

Create figures directory:

```bash
mkdir -p paper/figures
```

Generate figures using:

```bash
cd experiments
python generate_publication_figures.py
# Outputs saved to paper/figures/
```

## Submission Checklist

- [ ] All results filled in (TBD replaced with actual values)
- [ ] Figures generated and included
- [ ] Tables completed with experimental data
- [ ] References formatted correctly
- [ ] Supplementary material prepared
- [ ] Code repository linked
- [ ] Ethical considerations addressed
- [ ] Limitations discussed
- [ ] Author information anonymized (for review)

## Page Limit

ACL 2024: 8 pages main content + unlimited appendices

Current status: Main paper ~6 pages (estimated)

## Notes

- All results marked with "TBD" need to be filled with actual experimental data
- Run complete experimental pipeline first
- Generate all figures before final compilation
- Proofread for academic rigor
- Check mathematical notation consistency

