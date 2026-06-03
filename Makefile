# ============================================================
#  Makefile — Paper build automation
#  Usage:
#    make pdf       — compile English paper
#    make pdf_zh    — compile Chinese paper (requires XeLaTeX + ctex)
#    make figures   — regenerate all figures from v6 results
#    make tables    — regenerate all LaTeX table fragments
#    make all       — full pipeline: figures + tables + both PDFs
#    make clean     — remove build artifacts
# ============================================================

PROJECT_DIR := $(shell realpath ../machine_learning_for_crypto_v6)
SCRIPTS_DIR := $(shell pwd)/scripts
PAPER_DIR   := $(shell pwd)

.PHONY: all pdf pdf_zh figures tables clean

# ── Full pipeline ─────────────────────────────────────────────
all: figures tables pdf pdf_zh

# ── Compile PDFs ──────────────────────────────────────────────
pdf: tables
	cd $(PAPER_DIR) && latexmk -xelatex -interaction=nonstopmode main.tex

pdf_zh: tables
	cd $(PAPER_DIR) && latexmk -xelatex -interaction=nonstopmode main_zh.tex

# ── Generate figures from v6 checkpoints ──────────────────────
figures:
	@echo "=== Generating publication figures ==="
	cd $(PROJECT_DIR) && python $(SCRIPTS_DIR)/gen_figures.py
	cd $(PROJECT_DIR) && python $(SCRIPTS_DIR)/gen_ablation_fig.py

# ── Generate LaTeX table fragments ────────────────────────────
tables:
	@echo "=== Generating LaTeX tables ==="
	cd $(PROJECT_DIR) && python $(SCRIPTS_DIR)/gen_tables.py
	cd $(PROJECT_DIR) && python $(SCRIPTS_DIR)/gen_granger_table.py

# ── Re-run v6 evaluation (slow — only if results changed) ────
.PHONY: evaluate
evaluate:
	@echo "=== Re-running v6 evaluation ==="
	cd $(PROJECT_DIR) && python evaluate.py --config config.json --model cs_gated
	cd $(PROJECT_DIR) && python evaluate.py --config config.json --model lstm
	cd $(PROJECT_DIR) && python evaluate.py --config config.json --model tft
	cd $(PROJECT_DIR) && python evaluate.py --config config.json --compare-all

# ── Run Granger causality (captures stdout) ───────────────────
.PHONY: granger
granger:
	@echo "=== Running Granger causality test ==="
	cd $(PROJECT_DIR) && python granger_trump.py --config config.json \
	    > outputs/granger_output.txt 2>&1
	@echo "  Output saved to $(PROJECT_DIR)/outputs/granger_output.txt"

# ── Clean build artifacts ─────────────────────────────────────
clean:
	cd $(PAPER_DIR) && latexmk -C main.tex 2>/dev/null || true
	cd $(PAPER_DIR) && latexmk -C main_zh.tex 2>/dev/null || true
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg
	rm -f $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out $(PAPER_DIR)/*.toc
	rm -f $(PAPER_DIR)/*.fdb_latexmk $(PAPER_DIR)/*.fls
