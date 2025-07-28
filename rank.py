import random

class BinaryMaxHeap:
    """
    Helper max-heap class with external priorities list for rank-based sampling.
    """
    def __init__(self, priorities):
        self.priorities = priorities    # external list reference
        self.heap = []  # stores indices into priorities
        self.pos = {}   # maps index with position in heap array

    def push(self, idx):
        """Insert a new index into the heap."""
        self.heap.append(idx)
        i = len(self.heap) - 1
        self.pos[idx] = i
        self._sift_up(i)

    def update(self, idx):
        """After changing priorities[idx], restore heap order."""
        i = self.pos[idx]
        self._sift_up(i)
        self._sift_down(i)

    def _sift_up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.priorities[self.heap[i]] > self.priorities[self.heap[p]]:
                self._swap(i, p)
                i = p
            else:
                break

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            l, r, largest = 2*i + 1, 2*i + 2, i
            if l < n and self.priorities[self.heap[l]] > self.priorities[self.heap[largest]]:
                largest = l
            if r < n and self.priorities[self.heap[r]] > self.priorities[self.heap[largest]]:
                largest = r
            if largest != i:
                self._swap(i, largest)
                i = largest
            else:
                break

    def _swap(self, i, j):
        """Swap two positions in the heap."""
        hi, hj = self.heap[i], self.heap[j]
        self.heap[i], self.heap[j] = hj, hi
        self.pos[hi], self.pos[hj] = j, i

    def as_array(self):
        """
        Return the current heap array (approximately sorted by priority).
        """
        return self.heap[:]


class RankBasedHeapRM:
    """
    Rank-based prioritized replay memory.
    """
    def __init__(self, buffer_size, batch_size, epsilon=1e-6):
        self.buffer_size = buffer_size
        self.batch_size  = batch_size
        self.epsilon     = epsilon

        self.transitions = [None] * buffer_size
        self.priorities  = [0.0] * buffer_size
        self.heap        = BinaryMaxHeap(self.priorities)

        self.next_idx    = 0
        self._n_entries  = 0

    @property
    def num_entries(self):
        return self._n_entries

    def push(self, transition):
        """Add a new transition, initialize its priority, and maintain heap."""
        idx = self.next_idx
        # store the new transition
        self.transitions[idx] = transition

        # init its priority to the current max or 1.0 if empty.
        init_p = max(self.priorities[:self._n_entries]) if self._n_entries > 0 else 1.0
        self.priorities[idx] = init_p

        if self._n_entries < self.buffer_size:
            self.heap.push(idx)
            self._n_entries += 1
        else:
            self.heap.update(idx)

        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def sample(self):
        """
        Stratified sampling over ranked priorities.
        """
        N   = self._n_entries
        arr = self.heap.as_array()
        k   = self.batch_size
        seg = N // k

        batch, indices = [], []
        for i in range(k):
            start = i * seg
            end   = (i+1)*seg if i < k-1 else N
            if start >= end:
                j = random.randrange(N)
            else:
                j = random.randrange(start, end)
            idx = arr[j]
            batch.append(self.transitions[idx])
            indices.append(idx)
        return batch, indices

    def update(self, idx, td_error):
        self.priorities[idx] = abs(td_error) + self.epsilon
        self.heap.update(idx)