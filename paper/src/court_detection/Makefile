# LuaLaTeX + luatexja で日本語論文をビルドする。
#
#   make          : main.pdf を生成する（中間物は build/）
#   make watch    : latexmk -pvc で監視し、変更のたびに再構築する
#   make check    : ビルドして未定義参照・未定義引用・欠落ファイル・過大な Overfull を検査する
#   make clean    : 中間物のみ削除する
#   make distclean: 中間物と main.pdf を削除する
#
# watch のビューアは VIEW で選ぶ（既定は none）。例: make watch VIEW=pdf

MAIN := main
BUILD_DIR := build
LOG := $(BUILD_DIR)/$(MAIN).log
VIEW ?= none

SOURCES := $(MAIN).tex preamble.tex refs.bib latexmkrc \
	$(wildcard sections/*.tex) \
	$(wildcard appendices/*.tex) \
	$(wildcard figures/*.tex) \
	$(wildcard figures/generated/*.png) \
	$(wildcard figures/generated/*/*.png) \
	$(wildcard tables/*.tex) \
	$(wildcard results/generated/*.tex)

.PHONY: all watch check clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	@latexmk -lualatex -halt-on-error -file-line-error \
		-interaction=nonstopmode $(MAIN).tex

watch:
	@latexmk -lualatex -pvc -view=$(VIEW) -file-line-error \
		-interaction=nonstopmode $(MAIN).tex

check: $(MAIN).pdf
	@status=0; \
	if grep -nE "LaTeX Warning: (Reference|Citation) .* undefined" $(LOG); then \
		echo "check: 未定義の参照または引用があります" >&2; status=1; \
	fi; \
	if grep -nE "^! |Undefined control sequence|Emergency stop|TeX capacity exceeded" $(LOG); then \
		echo "check: TeX エラーがログに残っています" >&2; status=1; \
	fi; \
	if grep -nE "LaTeX Warning: File .* not found|No file .*\.(tex|png|pdf)" $(LOG); then \
		echo "check: 入力ファイルが見つかっていません" >&2; status=1; \
	fi; \
	if grep -nE "Overfull .hbox \(([4-9][0-9]|[0-9][0-9][0-9])\.|Overfull .vbox \(([4-9][0-9]|[0-9][0-9][0-9])\." $(LOG); then \
		echo "check: 40pt を超える Overfull box があります" >&2; status=1; \
	fi; \
	printf 'check: Overfull hbox '; grep -c 'Overfull .hbox' $(LOG) || true; \
	printf 'check: Overfull vbox '; grep -c 'Overfull .vbox' $(LOG) || true; \
	if [ $$status -eq 0 ]; then echo "check: OK"; fi; \
	exit $$status

clean:
	@latexmk -c $(MAIN).tex

distclean:
	@latexmk -C $(MAIN).tex
	@rm -f $(MAIN).pdf
