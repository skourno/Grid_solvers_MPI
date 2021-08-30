MPICC=mpicc
GCC=gcc
CFLAGS=-O3 -DPRINT_RESULTS
CONV=-DTEST_CONV
RINCPATH=-I/usr/include/mpi
SCIMPIPATH=-I/usr/include/openmpi
SCIMPILIBPATH=-L/usr/lib/openmpi
LIBFLAGS=-lm -lmpi

main:
	$(GCC) $(CFLAGS) $(SCIMPIPATH) $(SCIMPILIBPATH) -I. mpi_skeleton.c utils.c $(LIBFLAGS)
jacobi:
	$(GCC) $(CFLAGS) -DJACOBI $(CONV) $(SCIMPIPATH) $(SCIMPILIBPATH) -I. mpi_skeleton_jacobi.c utils.c $(LIBFLAGS)
gssor:
	$(GCC) $(CFLAGS) -DGSSOR  $(CONV) $(SCIMPIPATH) $(SCIMPILIBPATH) -I. mpi_skeleton_gssor.c utils.c $(LIBFLAGS)
redblacksor:
	$(GCC) $(CFLAGS) -DREDBLACK $(CONV) $(SCIMPIPATH) $(SCIMPILIBPATH) -I. mpi_skeleton_redblack.c utils.c $(LIBFLAGS)
#remote_jacobi:
#	$(MPICC) $(CFLAGS) -DJACOBI $(CONV) $(RINCPATH) mpi_skeleton_jacobi.c utils.c $(LIBFLAGS)

