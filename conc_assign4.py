import re
from mpi4py import MPI
import numpy as np
from copy import deepcopy
import time

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
    
    # TODO uncomment
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
    subimage = [ [0]*64 for col in range(64) ]

    # NUMPY
    ########################################################
    # subimages = np.ndarray(shape=(4,4))
    # for i,subimage in enumerate(subimages):
    #     subimage = np.ndarray(shape=(64,64), dtype=int)
    #     print(i,subimage.shape)
    ########################################################

    # Add padding?

    rank = comm.rank
    # DEBUG rank = 0

    # Populate subimage
    for current_row in range(64):
        for current_col in range(64):
            current_pixel = find_pixel_id(rank,current_row, current_col)
            subimage_row = subimage[current_row]
            subimage_row[current_col] = image[current_pixel]

    # print("rank ", rank, "subimage", subimage)
    
    # Apply filter and request data where needed
    final_subimage = [ [0]*64 for col in range(64) ]

    for current_row in range(64):
        for current_col in range(64):
            
            # print("rank", rank, "row", current_row, "col", current_col)
            
            for i in range(-4, 5):
                for j in range(-4,5):

                    # before i receive and block, probe to see if someone wants a message from me
                    status = MPI.Status()
                    probe = comm.iprobe(status = status)
                    if probe:
                        buf = np.zeros(1, dtype=int)
                        r = comm.Irecv(buf, source = status.Get_source(), tag = status.Get_tag()) # since we know there is a message, no waiting request
                        r.Wait(status)
                        print(rank, "recieved request with message", buf[0], "for pixel id", status.Get_tag(), "from", status.Get_source())
                        
                        # TODO fix: Asssuming this is always another node requesting a pixel 
                        [requested_pixel_rank, requested_pixel_row, requested_pixel_col] = find_subimage(buf[0])
                        p_value = subimage[requested_pixel_row][requested_pixel_col]
                        p = np.array([p_value], dtype=int)
                        comm.Isend(p, dest = status.Get_source(), tag = status.Get_tag())
                        print(rank, "sent message for pixel id", status.Get_tag(), "to", status.Get_source())

                    
                    if (current_row+i in range(64) and current_col+j in range(64)):
                        final_subimage[current_row][current_col] += subimage[current_row + i][current_col + j] * KERNEL[4 + i][4 + j]
                        # print("value without requests", final_subimage[current_row][current_col])

                    else:
                        requested_pixel_id = find_pixel_id(rank, current_row + i, current_col + j)
                        # print("req id", requested_pixel_id)
                        if requested_pixel_id == -1:
                            continue
                        [requested_pixel_rank, requested_pixel_row, requested_pixel_col] = find_subimage(requested_pixel_id)
                        # print("rank", rank, " found pixel", requested_pixel_id, "in rank", requested_pixel_rank)
                    
                        # This send should be sync?
                        r_id = np.array([requested_pixel_id], dtype=int)
                        comm.Isend(r_id, dest=requested_pixel_rank, tag = requested_pixel_id) 
                        # print(rank, "requested pixel ", requested_pixel_id, "from", requested_pixel_rank)

                        pixel_value = np.zeros(1, dtype=int)
                        req = comm.Irecv(pixel_value, source = requested_pixel_rank, tag = requested_pixel_id)
                        req.Wait()
                        print(rank, "reieved pixel id", requested_pixel_id, "from", requested_pixel_rank)
                        
                        final_subimage[current_row][current_col] += pixel_value[0] * KERNEL[4+i][4+j]

            if final_subimage[current_row][current_col] < 0:
                final_subimage[current_row][current_col] == 0
            elif final_subimage[current_row][current_col] > 255:
                final_subimage[current_row][current_col] == 255
        # TODO: how to ensure nodes stay alive? need them alive because others may request. BARRIER? 
    
    # Writing the output
    print("rank", rank, "started writing o/p")
    with open("output.ascii.pgm", "w") as f:
        
        f.write("P2")
        f.write("256 256")
        f.write("255")
        
        f.write(" ".join(final_image))
                    

    # Translate the pixel coordinates (current_row + i, current_col + j) to global coordinates
    # figure out which subimage it belongs to
    # create a request (we do need a send and a receive)
    ## OR can pre-calculate all the columns and rows each rank needs, but how to save them? can add three columns and rows of padding on each side?

    # if subimage(something) > max or < min, clip

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
    return global_id

#subimage_id would be the destination rank
def find_subimage(global_id):
    subimage_row = global_id // 16384
    subimage_col = global_id // 64 % 4
    subimage_id = (subimage_row*4 + subimage_col)
    row_within_subimage = global_id //256 % 64
    col_within_sub_image = global_id % 64
    return [subimage_id, row_within_subimage, col_within_sub_image]

main()



# Each node defines a 72 x 72 array with the 64 x 64 subimage in the middle
# Each node wants to fill out the empty spots
# Consider node 5
# For the top 4 rows we know that it is going to be receiving
# For the next 4 rows, except for first and last 4 pixels, we know that is it going to be sending (potentially to multiple nodes: 0,1,2,4,6,8,9,10)
# can have the node 5 receiving from ANY_SOURCE node because there are other nodes sending to node 5 but node 5 doesnt know when they will send
# use isend and irecv with the tag number equal global pixel id wanted

# Node 1 will have
# comm.Isend([requested_pixel_row, requested_pixel_col], dest=requested_pixel_rank, tag = requested_pixel_id)

# Node 5 will have
# while (???):
#   for requested_pixel_id in pixel_ids:
#       comm.Irecv([requested_pixel_row, requested_pixel_col], source=ANY_SOURCE, tag = requested_pixel_id)

# for loop over each

# for current_rank in range(16):
#     if comm.rank == current_rank:

#         find_pixel_you_want
#         comm.recv(pixel)

#     else:
#         while(??):
#             if irecv is not null
#                 comm.send(the_requested_pixels) 

#After this for loop, every subimage would have all the pixels it needs to calculate LoG