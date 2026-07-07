all: spectralstream_compression.pdf

%.pdf: %.tex
	pdflatex $<
	pdflatex $<

spectralstream_compression.pdf: spectralstream_compression.tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.pdf
