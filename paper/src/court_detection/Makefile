MAIN_EG := main
MAIN_JP := main_jp
BUILD_DIR := build
LATEXMK := $(shell command -v latexmk 2>/dev/null)

SHARED_SOURCES := preamble.tex refs.bib \
	$(wildcard appendices/*.tex) \
	$(wildcard figures/*.tex) \
	$(wildcard tables/*.tex)
EG_SOURCES := $(MAIN_EG).tex $(SHARED_SOURCES) $(wildcard sections/eg/*.tex)
JP_SOURCES := $(MAIN_JP).tex $(SHARED_SOURCES) $(wildcard sections/jp/*.tex)

.PHONY: all clean distclean watch watch-eg watch-jp

all: $(MAIN_EG).pdf $(MAIN_JP).pdf

ifdef LATEXMK
$(MAIN_EG).pdf: $(EG_SOURCES)
	@mkdir -p $(BUILD_DIR)
	latexmk -pdf -halt-on-error -file-line-error -interaction=nonstopmode \
		-outdir=$(BUILD_DIR) $(MAIN_EG).tex
	cp $(BUILD_DIR)/$(MAIN_EG).pdf $(MAIN_EG).pdf

$(MAIN_JP).pdf: $(JP_SOURCES)
	@mkdir -p $(BUILD_DIR)
	latexmk -lualatex -halt-on-error -file-line-error -interaction=nonstopmode \
		-outdir=$(BUILD_DIR) $(MAIN_JP).tex
	cp $(BUILD_DIR)/$(MAIN_JP).pdf $(MAIN_JP).pdf

watch: watch-eg

watch-eg:
	@mkdir -p $(BUILD_DIR)
	latexmk -pdf -pvc -file-line-error -interaction=nonstopmode \
		-outdir=$(BUILD_DIR) $(MAIN_EG).tex

watch-jp:
	@mkdir -p $(BUILD_DIR)
	latexmk -lualatex -pvc -file-line-error -interaction=nonstopmode \
		-outdir=$(BUILD_DIR) $(MAIN_JP).tex

clean:
	latexmk -c -outdir=$(BUILD_DIR) $(MAIN_EG).tex
	latexmk -c -lualatex -outdir=$(BUILD_DIR) $(MAIN_JP).tex

distclean:
	latexmk -C -outdir=$(BUILD_DIR) $(MAIN_EG).tex
	latexmk -C -lualatex -outdir=$(BUILD_DIR) $(MAIN_JP).tex
	$(RM) $(MAIN_EG).pdf $(MAIN_JP).pdf
else
PDFLATEX := pdflatex -halt-on-error -file-line-error -interaction=nonstopmode \
	-output-directory=$(BUILD_DIR)
LUALATEX := lualatex -halt-on-error -file-line-error -interaction=nonstopmode \
	-output-directory=$(BUILD_DIR)

$(MAIN_EG).pdf: $(EG_SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(PDFLATEX) $(MAIN_EG).tex
	cd $(BUILD_DIR) && BIBINPUTS=..: bibtex $(MAIN_EG)
	$(PDFLATEX) $(MAIN_EG).tex
	$(PDFLATEX) $(MAIN_EG).tex
	cp $(BUILD_DIR)/$(MAIN_EG).pdf $(MAIN_EG).pdf

$(MAIN_JP).pdf: $(JP_SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(LUALATEX) $(MAIN_JP).tex
	cd $(BUILD_DIR) && BIBINPUTS=..: bibtex $(MAIN_JP)
	$(LUALATEX) $(MAIN_JP).tex
	$(LUALATEX) $(MAIN_JP).tex
	cp $(BUILD_DIR)/$(MAIN_JP).pdf $(MAIN_JP).pdf

watch watch-eg watch-jp:
	@echo "watch requires latexmk" >&2
	@exit 1

clean:
	$(RM) -r $(BUILD_DIR)

distclean: clean
	$(RM) $(MAIN_EG).pdf $(MAIN_JP).pdf
endif
