import sys
from mpi4py import MPI
import numpy as np
from copy import deepcopy
import time
import trace 

PIXELS_PER_ROW = 64*64*4
comm = MPI.COMM_WORLD
KERNEL = [  
            [0, 0, 3, 2, 2, 2, 3, 0, 0], 
            [0, 2, 3, 5, 5, 5, 3, 2, 0],
            [3, 3, 5, 3, 0, 3, 5, 3, 3], 
            [2, 5, 3, -12, -23, -12, 3, 5, 2],
            [2, 5, 0, -23, -40, -23, 0, 5, 2], 
            [2, 5, 3, -12, -23, -12, 3, 5, 2],
            [3, 3, 5, 3, 0, 3, 5, 3, 3], 
            [0, 2, 3, 5, 5, 5, 3, 2, 0],
            [0, 0, 3, 2, 2, 2, 3, 0, 0]
         ]

def main():
    
    # Read the image
    if comm.rank == 0:
        with open("pepper.ascii.pgm", "r") as f:
            for _ in range(4):
                next(f)
            image = list(map(int,f.read().split()))
    else:
        image = None

    # Make image available to other nodes
    image = comm.bcast(image,0) 

    # Each node has a subimage
    subimage = np.zeros((72,72), dtype = int)
    final_subimage = np.zeros((64,64), dtype = int)

    rank = comm.rank
    # Populate subimage with core pixels
    for current_row in range(64):
        for current_col in range(64):
            current_pixel = find_pixel_id(rank,current_row, current_col)
            subimage[current_row+4, current_col+4] = image[current_pixel]

    # Populate subimage with non-core pixels on 2 steps
    # STEP 1: SEND/RECEIVE top/bottom/left/right
    # even row/even col and odd row/odd col 
    if ( (rank // 4) % 2 + (rank % 4) % 2 ) % 2 == 0:

        # Receive from the right (rank + 1) a 64x4 block then send, unless im the right column
        if (rank % 4 != 3) and within_image(rank+1):
            cols = comm.recv(source = rank + 1)
            subimage[4:68, 68:72] = cols

            cols = subimage[4:68, 64:68]
            comm.send(cols, dest = rank + 1)

        # Receive from the left (rank - 1) a 64x4 block then send, unless left column
        if (rank % 4 != 0) and within_image(rank-1):
            cols = comm.recv(source = rank - 1)
            subimage[4:68, 0:4] = cols

            cols = subimage[4:68, 4:8]
            comm.send(cols, dest = rank - 1)
        
        # Receive from the top (rank - 4) a 4x64 block then send, unless top row, included in within_image
        if within_image(rank-4):
            rows = comm.recv(source = rank - 4)
            subimage[0:4, 4:68] = rows

            rows = subimage[4:8, 4:68]
            comm.send(rows, dest = rank - 4)
        
        # Receive from the bottom (rank + 4) a 4x64 block then send, unless botttom row
        if within_image(rank+4):
            rows = comm.recv(source = rank + 4)
            subimage[68:72, 4:68] = rows


            rows = subimage[64:68, 4:68]
            comm.send(rows, dest = rank + 4)
    
    # even row/odd col and odd row/even col
    else:
        # Send to left (rank - 1) a 64x4 block then receive, unless im left column
        if (rank % 4 != 0) and within_image(rank-1):
            cols = subimage[4:68, 4:8]
            comm.send(cols, dest = rank - 1)

            cols = comm.recv(source = rank - 1)
            subimage[4:68, 0:4] = cols
        
        # Send to (rank + 1) a 64x4 block then receive
        if (rank % 4 != 3) and within_image(rank+1):
            cols = subimage[4:68, 64:68]
            comm.send(cols, dest = rank + 1)

            cols = comm.recv(source = rank + 1)
            subimage[4:68, 68:72] = cols
        
        # Send to (rank + 4) a 4x64 block then receive
        if within_image(rank+4):
            rows = subimage[64:68, 4:68]
            comm.send(rows, dest = rank + 4)

            rows = comm.recv(source = rank + 4)
            subimage[68:72, 4:68] = rows

        # Send to (rank - 4) a 4x64 block  then receive
        if within_image(rank - 4):
            rows = subimage[4:8, 4:68]
            comm.send(rows, dest = rank - 4)

            rows = comm.recv(source = rank - 4)
            subimage[0:4, 4:68] = rows

    # STEP 2: SEND/RECEIVE diagonals
    # Start with even rows
    if (rank // 4) % 2 == 0:
        
        # Bottom right
        if (rank % 4 != 3) and (rank // 4 != 3):
            subarr = comm.recv(source = rank + 5)
            subimage[68:72, 68:72] = subarr

            subarr = subimage[64:68, 64:68]
            comm.send(subarr, dest = rank + 5)

        # Bottom left
        if (rank % 4 != 0) and (rank // 4 != 3):
            subarr = comm.recv(source = rank + 3)
            subimage[68:72, 0:4] = subarr

            subarr = subimage[64:68, 4:8]
            comm.send(subarr, dest = rank + 3)

        # Top right
        if (rank % 4 != 3) and (rank // 4 != 0):
            subarr = comm.recv(source = rank - 3)
            subimage[0:4, 68:72] = subarr

            subarr = subimage[4:8, 64:68]
            comm.send(subarr, dest = rank - 3)

        # Top left
        if (rank % 4 != 0) and (rank // 4 != 0):
            subarr = comm.recv(source = rank - 5)
            subimage[0:4, 0:4] = subarr

            subarr = subimage[4:8, 4:8]
            comm.send(subarr, dest = rank - 5)
    
    # Odd rows
    else:
        # Top left
        if (rank % 4 != 0) and (rank // 4 != 0):
            subarr = subimage[4:8, 4:8]
            comm.send(subarr, dest = rank - 5)

            subarr = comm.recv(source = rank - 5)
            subimage[0:4, 0:4] = subarr

        # Top right
        if (rank % 4 != 3) and (rank // 4 != 0):
            subarr = subimage[4:8, 64:68]
            comm.send(subarr, dest = rank - 3)

            subarr = comm.recv(source = rank - 3)
            subimage[0:4, 68:72] = subarr
        
        # Bottom left
        if (rank % 4 != 0) and (rank // 4 != 3):
            subarr = subimage[64:68, 4:8]
            comm.send(subarr, dest = rank + 3)
            
            subarr = comm.recv(source = rank + 3)
            subimage[68:72, 0:4] = subarr      
        
        # Bottom right
        if (rank % 4 != 3) and (rank // 4 != 3):
            subarr = subimage[64:68, 64:68]
            comm.send(subarr, dest = rank + 5)

            subarr = comm.recv(source = rank + 5)
            subimage[68:72, 68:72] = subarr

    # # For debugging
    # for i in range(16):
    #     if i != rank:
    #         time.sleep(2)
    #     else:
    #         print("rank", rank, "subimage", subimage)        

# At this point all nodes should have all the values they need, compute the filter
    for current_row in range(4,68):
        for current_col in range(4,68):  
        
            # For each pixel, perform 81 multiplications
            for i in range(-4, 5):
                for j in range(-4,5):
                    final_subimage[current_row-4, current_col-4] += subimage[current_row + i, current_col + j] * KERNEL[i+4][j+4]
            
            # After applying kernel on a pixel, clip the value if beyond limits
            current_value = final_subimage[current_row-4, current_col-4]
            if current_value < 0:
                final_subimage[current_row-4, current_col-4] = 0
            elif current_value > 255:
                final_subimage[current_row-4, current_col-4] = 255

    final_image = np.zeros((256,256), dtype = int)
    # Gather the image in one array
    if rank != 0:
        comm.send(final_subimage, dest=0)

    else:
        final_image[0:64, 0:64] = final_subimage

        for i in range(1,16):
            subimage = comm.recv(source=i)
            
            row_begin = (i // 4) * 64
            row_end = row_begin + 64
            col_begin = (i % 4) * 64
            col_end = col_begin + 64

            print("rank", i, "values:", row_begin, row_end, col_begin, col_end)

            final_image[row_begin:row_end, col_begin:col_end] = subimage
    
            # Write the output
            with open("output.ascii.pgm", "w") as f:
                
                f.write("P2\n")
                f.write("256 256\n")
                f.write("255\n")
                with np.printoptions(threshold=np.inf):
                    x = str(list(final_image.reshape(-1)))
                    f.write( x.replace(",", "").replace("[", "").replace("]", ""))

# subimage_index would be the requester's rank. should pass (current_row + i) as current_row
def find_pixel_id(subimage_index, current_row, current_col):
   
    if current_row < 0 or current_col < 0:
        return -1

    if subimage_index%4 == 3 and current_col>63:
        return -1
   
    pixels_prev_rows = subimage_index//4 * (64*64*4) + (current_row)*256
    pixels_curr_row_prev_subimages = (subimage_index%4)*64
    pixels_curr_row_same_subimage = current_col

    global_id = pixels_prev_rows + pixels_curr_row_prev_subimages + pixels_curr_row_same_subimage
    if global_id > 65535:
        return -1
    return global_id

#subimage_id would be the destination rank
def find_subimage(global_id):
    subimage_row = global_id // 16384
    subimage_col = global_id // 64 % 4
    subimage_id = (subimage_row*4 + subimage_col)
    row_within_subimage = global_id //256 % 64
    col_within_sub_image = global_id % 64
    return [subimage_id, row_within_subimage, col_within_sub_image]

def within_image(rank):
    return rank <= 15 and rank >=0

main()
