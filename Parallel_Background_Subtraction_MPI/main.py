from mpi4py import MPI
from mpi_processing.background_subtraction import parallel_background_subtraction
from utils.image_utils import extract_frames_from_gif

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        extract_frames_from_gif("Background-Subtraction-Tutorial_merged.gif", "input/")

    comm.Barrier()
    parallel_background_subtraction(comm, rank, size)

if __name__ == "__main__":
    main()
