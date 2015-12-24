.SUFFIXES: .c .o .swg

include makefile.inc

CC=gcc


ALLDIRS=yael progs

all: $(addprefix compiledir.,$(ALLDIRS))
	python test/py/test_gmm_w.py

clean: $(addprefix cleandir.,$(ALLDIRS))



compiledir.%:
	cd $(subst compiledir.,,$@); make

cleandir.%:
	cd $(subst cleandir.,,$@); make clean

