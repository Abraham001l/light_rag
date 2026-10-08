import math
import random
import heapq
import torch
import torch.nn.functional as F

class hnsw_database:
    def __init__(self, max_neighbors=16, ef_construction=500):
        self.nodes = []
        self.max_neighbors = max_neighbors
        self.ef_construction = ef_construction
        self.max_neighbors0 = max_neighbors*2
        self.level_multiplier = 1 / math.log(max_neighbors)
        self.cur_max_level = -1
        self.entry_point = None

    def add_node(self, tensor):
        if (self.entry_point is None):
            # setting up first node as entry point
            max_level = self.get_random_level()
            self.entry_point = vnode(tensor, max_level)
            self.nodes.append(self.entry_point)
            self.cur_max_level = max_level
        else:
            max_level = self.get_random_level()
            new_node = vnode(tensor, max_level)

            # db's to track exploration of neighbors (set, min-heap, max-heap)
            visited = set()
            next_in_line = [] 
            closest_nodes = []

            # start exploration from entry point
            cur_node = self.entry_point
            visited.add(cur_node)
            heapq.heappush(next_in_line, (0, cur_node))

            # exploration loop
            while next_in_line:
                cur_node = heapq.heappop(next_in_line)[1]

                # claculate distance and add to closest_nodes
                cur_dist = self.get_distance(new_node, cur_node)
                heapq.heappush_max(closest_nodes, (cur_dist, cur_node))
                

            


    def get_random_level(self):
        r = random.random()
        assigned_level = math.floor(-math.log(r)*self.level_multiplier)
        return max(0, assigned_level)

    def get_distance(self, node1, node2):
        return 1 - torch.dot(node1.normed_tensor, node2.normed_tensor).item()

    
class vnode:
    def __init__(self, tensor=None, max_layer=0):
        self.tensor = tensor
        self.normed_tensor = F.normalize(tensor, p=2, dim=0) if tensor is not None else None
        self.neighbors_by_layer = []*(max_layer+1)

    def add_neighbor(self, neighbor_node, layer):
        None