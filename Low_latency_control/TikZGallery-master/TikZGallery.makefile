ALL_FIGURE_NAMES=$(shell cat TikZGallery.figlist)
ALL_FIGURES=$(ALL_FIGURE_NAMES:%=%.pdf)

allimages: $(ALL_FIGURES)
	@echo All images exist now. Use make -B to re-generate them.

FORCEREMAKE:

include $(ALL_FIGURE_NAMES:%=%.dep)

%.dep:
	mkdir -p "$(dir $@)"
	touch "$@" # will be filled later.

Figs/pdf/pgf-o-InvertedPendulum.pdf: Figs/Src/InvertedPendulum.tikz
Figs/pdf/pgf-o-InvertedPendulum.pdf: 
	pdflatex -halt-on-error -interaction=batchmode -jobname "Figs/pdf/pgf-o-InvertedPendulum" "\newif\ifpgfnotitlebg\pgfnotitlebgtrue\input{TikZGallery}"

Figs/pdf/pgf-o-InvertedPendulum.pdf: Figs/pdf/pgf-o-InvertedPendulum.md5
Figs/pdf/pgf-o-VehicleControl.pdf: Figs/Src/VehicleControl.tikz
Figs/pdf/pgf-o-VehicleControl.pdf: 
	pdflatex -halt-on-error -interaction=batchmode -jobname "Figs/pdf/pgf-o-VehicleControl" "\newif\ifpgfnotitlebg\pgfnotitlebgtrue\input{TikZGallery}"

Figs/pdf/pgf-o-VehicleControl.pdf: Figs/pdf/pgf-o-VehicleControl.md5
Figs/pdf/pgf-o-PCATransform.pdf: Figs/Src/PCATransform.tikz
Figs/pdf/pgf-o-PCATransform.pdf: 
	pdflatex -halt-on-error -interaction=batchmode -jobname "Figs/pdf/pgf-o-PCATransform" "\newif\ifpgfnotitlebg\pgfnotitlebgtrue\input{TikZGallery}"

Figs/pdf/pgf-o-PCATransform.pdf: Figs/pdf/pgf-o-PCATransform.md5
Figs/pdf/pgf-o-SieveofEratosthenes.pdf: Figs/Src/SieveofEratosthenes.tikz
Figs/pdf/pgf-o-SieveofEratosthenes.pdf: 
	pdflatex -halt-on-error -interaction=batchmode -jobname "Figs/pdf/pgf-o-SieveofEratosthenes" "\newif\ifpgfnotitlebg\pgfnotitlebgtrue\input{TikZGallery}"

Figs/pdf/pgf-o-SieveofEratosthenes.pdf: Figs/pdf/pgf-o-SieveofEratosthenes.md5
