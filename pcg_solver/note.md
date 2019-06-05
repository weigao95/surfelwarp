### A note on the format of the matrix

To represent the sparse matix, the Bin Blocked Compressed Row Storage is used.  The bin length is always the warp size = 32, while the block size depends on applications. In the following text, we use block = 6 as it is implemented and tested in this project.

### Notations:
1. blk = block size, 6 by default
2. Nblk, the number of blocks along each matrix dimension. The matrix is a symmetric, positive definite matrix in the size of (blk x Nblk). 
3. bin_size = 32. The continuous bin\_size (32) rows are collected in a bin for concurrent CUDA threaded access. 
3. Nbin, the number of bins used for the representation of the matrix, the value is divUp(Nblk x blk, bin_size)

### Data structures:
The program uses arrays to represent the matrix

1. Flattened data array(A\_data): Flatten all the elements of the matrix over each bin using row major. For each bin, there a bin_size(32) x bin_length continuous memory. Inside the bin, the flatten is colume major, i.e., if the element matrix(r, c) is located at loc(r, c), then the matrix(r, c + 1) is located in loc(r, c) + bin_size(32) in flatten memory.

2. Data Row Offset(rowptr): in the size of (Nbin + 1) x bin_size,. For each row r, rowptr[r] is the offset of the first element in this row, i.e., A\_data[rowptr[r]] is the value of the first element in this row. For rows in a bin, the rowptr array is continuous, i.e., rowptr[r + 1] = rowptr[r] + 1.

3. The column index(colptr): in classic CSR format, colptr should be in the same size and layout as A\_data, as each element should know its column. In blocked CSR format, the matrix has blocked structure. 
To be more specific, for each row the number of elements in this row must be a multiple of blk. Suppose there is (num\_row\_blks(r) * blk) elements in row r, we only need to store num\_row_blks(r) element in the colptr array. 
In Bin Blocked CSR format, suppose the bin length of bin(r) is num\_bin\_blks(bin(r)), for all rows in this bin, store num\_bin\_blks(bin(r)) column offsets, where the invalid (padded) blocks are set to invalid column index. For each row, consecutive block columns follow: loc(row, blk_col + 1) = loc(row, blk\_col ) + bin\_size

### Algorithms:
####The algorihtm to compute the rowptr is:
Input: An array of (blk_row, blk\_col) pairs. 
Procedure:
1. sort the pairs according to blk_row and compaction
2. for each bin i, compute the bin_length: the maximum for row in the range of [bin\_size * i, bin\_size * (i + 1)). 
3. for each bin i, the number of elements in this bin is bin\_size x bin\_length, do a prefix sum to obtain an array bin_rowptr in the size of (Nbin + 1), where bin\_rowptr[i] is the offset of the first element in this bin.
4. Compute rowptr from bin_rowptr: rowptr[r] = bin\_rowptr[bin(r)] + inbin\_offset, where bin(r) is the bin that this row lives in, inbin\_offset is the offset of this row in the bin. 

Note the tranform betwenn blk_row and column to actual row


####The algorithm to compute the colptr is:
Input: rowptr array
Output: colptr array, which in pre-allocated and set to null values
Procudure:
forall row index r in range(blk x Nblk):
1. blk\_row = r / N, inblk_offset = r % N, bin\_row = r / bin\_size, inbin\_offset = r % bin\_size, data\_offset = rowptr(r)
2. col\_idx\_offset = (data\_offset - inbin\_offset) / N + inbin\_offset //data_offset - inbin\_offset is the data offset of the first element in this bin, its col\_idx\_offset should be  (data\_offset - inbin\_offset) / N. Thus, the col\_idx\_offset of this element shoule be (data\_offset - inbin\_offset) / N + inbin\_offset
4. Iterates over all block colums in this row, colptr(col\_idx\_offset += 32) = 6 * column