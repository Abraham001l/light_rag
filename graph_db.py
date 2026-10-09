import pickle

class graph:
    def __init__(self):
        self.hashmap_to_nodes = {}

    def add_node(self, key, description):
        if key in self.hashmap_to_nodes.keys():
            old_description = self.hashmap_to_nodes[key].description
            self.hashmap_to_nodes[key].update_val(old_description + "\n" + description)
            return False
        else:
            self.hashmap_to_nodes[key] = gnode(description)
            return True

    def add_neighbors_to_node(self, my_key, neighbors_keys):
        my_node = self.hashmap_to_nodes[my_key]
        for neighbor_key in neighbors_keys:
            my_node.add_neighbor(neighbor_key)
            neighbor_node = self.hashmap_to_nodes[neighbor_key]
            neighbor_node.add_neighbor(my_key)

    def get_node(self, key):
        return self.hashmap_to_nodes.get(key, None)

    def get_neighbors_of_node(self, key):
        node = self.hashmap_to_nodes.get(key, None)
        if node is not None:
            return node.neighbors
        else:
            return None

    # AI CODE -------------------------------------------
    def save(self, filepath="semantic_graph.pkl"):
        # Extract the raw data into a clean dictionary
        export_data = {}
        for key, node in self.hashmap_to_nodes.items():
            export_data[key] = {
                "description": node.description,
                "neighbors": node.neighbors
            }
            
        with open(filepath, "wb") as f:
            pickle.dump(export_data, f)
            
        print(f"Successfully saved {len(self.hashmap_to_nodes)} nodes to {filepath}")

    @classmethod
    def load(cls, filepath="semantic_graph.pkl"):
        with open(filepath, "rb") as f:
            import_data = pickle.load(f)
            
        # Re-initialize a blank graph
        new_graph = cls()
        
        # Rebuild the nodes and attach them to the hashmap
        for key, data in import_data.items():
            new_node = gnode(description=data["description"])
            new_node.neighbors = data["neighbors"]
            new_graph.hashmap_to_nodes[key] = new_node
            
        print(f"Successfully loaded {len(new_graph.hashmap_to_nodes)} nodes from {filepath}")
        return new_graph
    # AI CODE -------------------------------------------

class gnode:
    def __init__(self, description=None, neighbors=None):
        self.description = description
        self.neighbors = neighbors if neighbors is not None else set()

    def add_neighbor(self, neighbor_key):
        self.neighbors.add(neighbor_key)

    def update_val(self, new_val):
        self.description = new_val
    

    