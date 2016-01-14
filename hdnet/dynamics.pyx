# @cython.boundscheck(False)
# @cython.wraparound(False)
def heap_converge(
        np.ndarray[long, ndim=1, mode="c"] x,
        np.ndarray[double, ndim=2, mode="c"] J, double theta):

    cdef int n
    cdef float ff
    cdef int pos

    n = x.shape[0]

    cdef np.ndarray[int, ndim=1, mode="c"] heap = np.zeros((n,), dtype=np.int32)
    cdef np.ndarray[int, ndim=2, mode="c"] data = np.zeros((n, 2), dtype=np.int32)

    for pos in range(n):
		ff = np.dot(J[pos, :], x) - theta[pos]
        data[pos, 0] = pos
        data[pos, 1] = int(10000 * np.abs(ff))  # 10000 is hmmmm
		if ff > 0:
			data[pos, 2] = 1
		else:
			data[pos, 2] = 0
        heap[pos] = pos

    cdef Heap h = Heap(heap, data)

    while True:
        pos = h.pop()
        if pos < 0:
            break

		if x[pos] != data[pos, 2]:
			x[pos] = data[pos, 2]
			ff = np.dot(J[pos, :], x) - theta[pos]
            data[pos, 1] = int(10000 * np.abs(ff))
			if ff > 0:
				data[pos, 2] = 1
			else:
				data[pos, 2] = 0
            h.siftup(data[pos, 0])


cdef class Heap:
    """ Max-heap. """
    cdef public np.ndarray data
    cdef public np.ndarray heap
    cdef int N


    def __init__(Heap self,
            np.ndarray[int, ndim=1, mode="c"] heap,
            np.ndarray[int, ndim=2, mode="c"] data):

        cdef int i
        cdef int N = data.shape[0]
        assert heap.shape[0] == N, "Heap and data have to have same length"
        assert data.shape[1] > 1, "Data need at least two columns"
        self.data = data
        self.heap = heap
        self.N = N

        for i in range(N // 2 - 1, -1, -1):
            self.siftup(i)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void siftdown(Heap self, int startpos, int pos):
        cdef np.ndarray[int, ndim=1, mode="c"] heap
        cdef np.ndarray[int, ndim=2, mode="c"] data
        heap = self.heap
        data = self.data

        cdef int newidx, newitem, newval
        cdef int parentpos, parentidx, parentval

        newidx = heap[pos]
        newval = data[newidx, 0]
        # Follow the path to the root, moving parents down
        # until finding a place newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parentidx = heap[parentpos]
            parentval = data[parentidx, 0]
            if newval > parentval:
                heap[pos] = parentidx
                data[parentidx, 1] = pos
                pos = parentpos
                continue
            break

        heap[pos] = newidx
        data[newidx, 1] = pos


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void siftup(Heap self, int pos):
        cdef np.ndarray[int, ndim=1, mode="c"] heap
        cdef np.ndarray[int, ndim=2, mode="c"] data
        heap = self.heap
        data = self.data

        cdef int idx, newidx, newitem, newval
        cdef int startpos, childpos, rightpos

        startpos = pos
        newidx = heap[pos]
        newval = data[newidx, 0]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < self.N:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < self.N and not data[heap[childpos], 0] > data[heap[rightpos], 0]:
                childpos = rightpos

            # Move the smaller child up.
            idx = heap[childpos]
            heap[pos] = idx
            data[idx, 1] = pos
            pos = childpos
            childpos = 2*pos + 1

        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = newidx
        data[newidx, 1] = pos
        self.siftdown(startpos, pos)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int pop(Heap self):
        """ Pop maximal item.

        Pop the largest item off the heap, maintaining the 
        heap invariant.
        """
        cdef np.ndarray[int, ndim=1, mode="c"] heap
        cdef np.ndarray[int, ndim=2, mode="c"] data
        heap = self.heap
        data = self.data

        cdef int idx, lastidx, lastval, returnval

        if self.N == 0:
            return -1

        self.N -= 1
        lastidx = heap[self.N]
        lastval = data[lastidx, 0]

        if self.N:
            idx = heap[0]
            returnval = data[idx, 0]
            heap[0] = lastidx
            data[lastidx, 1] = 0
            self.siftup(0)
        else:
            returnval = lastval

        return returnval
