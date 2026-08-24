MAIN := main
BUILD_DIR := build
LATEXMK := $(shell command -v latexmk 2>/dev/null)
SOURCES := $(MAIN).tex preamble.tex refs.bib \
	$(wildcard sections/*/*.tex) \
	$(wildcard appendices/*.tex) \
	$(wildcard figures/*.tex) \
	$(wildcard tables/*.tex)

.PHONY: all clean distclean watch

all: $(MAIN).pdf

ifdef LATEXMK
$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	latexmk -pdf -halt-on-error -file-line-error -interaction=nonstopmode \
		-outdir=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $(MAIN).pdf

watch:
	@mkdir -p $(BUILD_DIR)
	latexmk -pdf -pvc -file-line-error -interaction=nonstopmode \
		-outdir=$(BUILD_DIR) $(MAIN).tex

clean:
	latexmk -c -outdir=$(BUILD_DIR) $(MAIN).tex

distclean:
	latexmk -C -outdir=$(BUILD_DIR) $(MAIN).tex
	$(RM) $(MAIN).pdf
else
PDFLATEX := pdflatex -halt-on-error -file-line-error -interaction=nonstopmode \
	-output-directory=$(BUILD_DIR)

$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(PDFLATEX) $(MAIN).tex
	cd $(BUILD_DIR) && BIBINPUTS=..: bibtex $(MAIN)
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $(MAIN).pdf

watch:
	@echo "watch requires latexmk" >&2
	@exit 1

clean:
	$(RM) -r $(BUILD_DIR)

distclean: clean
	$(RM) $(MAIN).pdf
endif
