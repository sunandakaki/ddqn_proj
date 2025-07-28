import numpy as np

class SumTreeRM():
    """
    Proportional priority replay memory using a sum-tree data structure.
    """
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tree = np.zeros(2*self.buffer_size-1)  # Tree array: size = 2*buffer_size - 1
        self.transitions = [None]* self.buffer_size
        self.pointer_idx = 0    # next write position
        self.num_entries = 0    # number of filled slots
    
    def push(self, transition, priority=1):
        """
        Add a transition with given priority (default=1 for new entries).
        """
        sumtree_idx = self.pointer_idx + (self.buffer_size - 1) # Leaf index in sum-tree
        self.transitions[self.pointer_idx] = transition
        self.update(priority, sumtree_idx)
        self.pointer_idx = (self.pointer_idx + 1) % self.buffer_size
        self.num_entries = min(self.num_entries + 1, self.buffer_size)  # Track number of stored entries
    
    def update(self, priority, sumtree_idx):
        """
        Update the priority at leaf 'sumtree_idx', and propagate the change upward.
        """
        change = priority - self.tree[sumtree_idx]
        self.tree[sumtree_idx] = priority
        while sumtree_idx != 0:
            sumtree_idx = (sumtree_idx - 1)//2
            self.tree[sumtree_idx] += change

    def get_transition(self, value):
        """
        Traverse the tree to find the leaf index corresponding to 'value'.
        """
        idx = 0
        while idx < (self.buffer_size - 1):
            left_child_idx = 2*idx + 1
            right_child_idx = 2*idx + 2
            if value <= self.tree[left_child_idx]:
                idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                idx = right_child_idx
        trans_idx = idx - (self.buffer_size - 1)
        return idx, self.tree[idx], self.transitions[trans_idx]

    def sample(self):
        """
        Sample a minibatch of transitions with stratified proportional priorities.
        """
        batch = []
        index = []
        parition = self.tot_priority()/self.batch_size
        for i in range(self.batch_size):
            value = np.random.uniform(i*parition, (i+1)*parition)
            sampled_idx, _, sampled_trans = self.get_transition(value)
            batch.append(sampled_trans)
            index.append(sampled_idx)
            
        return batch, index
    
    def tot_priority(self):
        return self.tree[0]