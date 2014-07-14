import numpy as np
from collections import defaultdict


G = 44923.53
deltas = np.array([[1, 1, 1], [-1, 1, 1],
                   [1, -1, 1], [1, 1, -1],
                   [-1, -1, 1], [-1, 1, -1],
                   [1, -1, -1], [-1, -1, -1]])


class oct_tree():
    def __init__(self, side, center = np.array((0, 0, 0)), COM = np.array((0, 0, 0)), mass = 0):
        self.COM = COM
        self.mass = mass
        self.center = center
        self.side = side
        self.branches = [None] * 8
        self.childless = True

    def find_place(self, pos):
        signs = np.sign(pos-self.center)
        index = 0
        while not np.array_equal(signs, deltas[index]):
            index += 1
        return index
 
    def insert(self, pos, mass):
        if self.childless and self.mass > 0:
            insertion_list = [[pos, mass], [self.COM, self.mass]]
        else:
            insertion_list = [[pos, mass]]
        if self.childless:
            self.childless = False

        for p, m in insertion_list:
            index = self.find_place(p)
            if self.branches[index] == None:
                self.branches[index] = oct_tree(self.side/2.0, center = self.center + self.side/4.0*deltas[index], COM = p, mass = m)
            else:
                self.branches[index].insert(p, m)
        self.COM = (self.mass*self.COM + mass*pos) / (self.mass+mass)
        self.mass += mass


def potential(pos, tree):
    d = np.linalg.norm(tree.COM-pos)
    if tree.childless:
        return -G*tree.mass/d
    theta = tree.side / d
    if theta < 0.5:
        return -G*tree.mass / d
    else:
        sum_ = 0.0
        for branch in tree.branches:
            if branch != None:
                sum_ += potential(pos, branch)
        return sum_
