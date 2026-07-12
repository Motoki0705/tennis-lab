# =====================================================================
#  paper/Makefile
#
#  Uses latexmk when available (incremental, gets the number of passes
#  right by itself). Falls back to an explicit pdflatex/bibtex/pdflatex
#  x2 sequence, which is all this machine currently has installed.
#
#    make            build main.pdf
#    make watch      rebuild on save (requires latexmk)
#    make clean      remove build artifacts, keep main.pdf
#    make distclean  remove build artifacts and main.pdf
# =====================================================================

MAIN     := main
BUILDDIR := build
LATEXMK  := $(shell command -v latexmk 2>/dev/null)

SOURCES := $(MAIN).tex preamble.tex $(wildcard chapters/*.tex) refs.bib

.PHONY: all watch clean distclean
all: $(MAIN).pdf

ifdef LATEXMK

$(MAIN).pdf: $(SOURCES)
	latexmk -pdf -halt-on-error -interaction=nonstopmode \
	        -outdir=$(BUILDDIR) $(MAIN).tex
	@cp $(BUILDDIR)/$(MAIN).pdf $(MAIN).pdf

watch:
	latexmk -pdf -pvc -interaction=nonstopmode -outdir=$(BUILDDIR) $(MAIN).tex

clean:
	latexmk -c -outdir=$(BUILDDIR) $(MAIN).tex

distclean:
	latexmk -C -outdir=$(BUILDDIR) $(MAIN).tex
	@rm -rf $(BUILDDIR) $(MAIN).pdf

else

# --- Fallback: no latexmk installed ---------------------------------
PDFLATEX := pdflatex -halt-on-error -interaction=nonstopmode \
            -output-directory=$(BUILDDIR)

$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(BUILDDIR)
	$(PDFLATEX) $(MAIN).tex
	@# bibtex runs inside BUILDDIR and must see refs.bib via BIBINPUTS
	cd $(BUILDDIR) && BIBINPUTS=..: bibtex $(MAIN) || true
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex
	@cp $(BUILDDIR)/$(MAIN).pdf $(MAIN).pdf

watch:
	@echo "watch requires latexmk. Install it with:"
	@echo "  sudo apt install latexmk"
	@exit 1

clean:
	@rm -rf $(BUILDDIR)

distclean: clean
	@rm -f $(MAIN).pdf

endif
