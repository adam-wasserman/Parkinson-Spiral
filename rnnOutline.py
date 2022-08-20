import numpy as np

class GRU:

    def __init__(self, data):
        self.data = data #data = [[0,0,0,0,0,0], [0,0,0,0,0,0]] Multi-dimensional array with 6 inputs at a given time
        self.ct = [0, 0, 0, 0, 0, 0]  # TODO figure out if we aren't going to use CH 1
        self.ht = [0, 0, 0, 0, 0, 0]

    def GRUCell(prev_ct, prev_ht, input):
        combine = prev_ht + input


    def run(self):
        for input in self.data:
            self.ct,self.ht = self.GRUCell(self.ct, self.ht, input)