from mpi4py import MPI
import numpy as np
import cv2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NUM_OF_CENTERS = 2

# The K Mean function
def run_kmeans(image, k):
    pixel_vals = image.reshape((-1, 3))

    # This converts the type to float
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # Get the image in the normal shape
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image


if rank == 0:
    # Read in the image
    image = cv2.imread('img.jpg')

    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    chunks = np.array_split(image, size - 1, 0)

    for s_index in range(0, size - 1):
        comm.send(chunks[s_index], dest=s_index + 1)

else:
    image = comm.recv(source=0)
    segmented_image = run_kmeans(image, NUM_OF_CENTERS)
    comm.send(segmented_image, dest=0)

if rank == 0:
    image = comm.recv(source=1)

    for k in range(2, size):
        image = cv2.vconcat([image, comm.recv(source=k)])

    # Write the image to result.png
    cv2.imwrite('result.png', image)
