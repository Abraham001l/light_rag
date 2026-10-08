
class graph:
    def __init__(self):
        self.hashmap_to_nodes = {}

    def add_node(self, key, val):
        if key in self.hashmap_to_nodes.keys():
            # TODO: add special handling
            None
        else:
            self.hashmap_to_nodes[key] = node(val)

    def add_neighbors_to_node(self, my_key, neighbors_keys):
        my_node = self.hashmap_to_nodes[my_key]
        for neighbor_key in neighbors_keys:
            my_node.add_neighbor(neighbor_key)
            neighbor_node = self.hashmap_to_nodes[neighbor_key]
            neighbor_node.add_neighbor(my_key)

    

class node:
    def __init__(self, val=None, neighbors=set()):
        self.val = val
        self.neighbors = neighbors

    def add_neighbor(self, neighbor_key):
        self.neighbors.add(neighbor_key)
    

    