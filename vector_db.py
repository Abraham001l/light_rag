import math
import random
import heapq
import torch
import torch.nn.functional as F

class hnsw_database:
    def __init__(self, max_neighbors=16, ef_construction=50):
        self.nodes = []
        self.max_neighbors = max_neighbors
        self.ef_construction = ef_construction
        self.max_neighbors0 = max_neighbors*2
        self.level_multiplier = 1 / math.log(max_neighbors)
        self.cur_max_level = -1
        self.entry_point = None

    def add_node(self, tensor):
        if self.entry_point is None:
            # setting up first node as entry point
            max_level = self.get_random_level()
            self.entry_point = vnode(tensor, max_level)
            self.nodes.append(self.entry_point)
            self.cur_max_level = max_level
            return self.entry_point
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
            cur_layer = math.inf
            while next_in_line:
                cur_node = heapq.heappop(next_in_line)[1]
                cur_dist = self.get_distance(new_node, cur_node)
                cur_layer = min(cur_layer, cur_node.max_layer)

                # divide exploration into either fast or deep depending on if level is part of new node's levels
                if cur_layer > new_node.max_layer:
                    level_min_dist = cur_dist
                    level_min_node = cur_node
                    changed = False
                    # explore only cur_node's neighbors and move down
                    for neighbor_node in cur_node.neighbors_by_layer[cur_layer]:
                        if neighbor_node not in visited:
                            visited.add(neighbor_node)
                            neighbor_dist = self.get_distance(new_node, neighbor_node)
                            if neighbor_dist < level_min_dist:
                                level_min_dist = neighbor_dist
                                level_min_node = neighbor_node
                                changed = True

                    # traversing down through closest node
                    next_in_line.clear()
                    heapq.heappush(next_in_line, (level_min_dist, level_min_node))
                    
                    if not changed:
                        cur_layer -= 1
                        if cur_layer < 0:
                            break
                        # only clear visited and closest_nodes when moving to a new layer!
                        visited.clear()
                        closest_nodes.clear()
                        heapq.heappush(closest_nodes, (-level_min_dist, level_min_node))
                else:
                    # if we skipped fast drop, seed the closest_nodes heap
                    if len(closest_nodes) == 0:
                        heapq.heappush(closest_nodes, (-cur_dist, cur_node))

                    # explore cur_node's neighbors
                    for neighbor_node in cur_node.neighbors_by_layer[cur_layer]:
                        if neighbor_node not in visited:
                            visited.add(neighbor_node)
                            neighbor_dist = self.get_distance(new_node, neighbor_node)
                            heapq.heappush(next_in_line, (neighbor_dist, neighbor_node))
                            if len(closest_nodes) < self.ef_construction:
                                heapq.heappush(closest_nodes, (-neighbor_dist, neighbor_node))
                            else:
                                if -neighbor_dist > closest_nodes[0][0]:
                                    heapq.heapreplace(closest_nodes, (-neighbor_dist, neighbor_node))

                    # determine if we should continue exploring neighbors or move down a layer
                    if  len(next_in_line) == 0 or next_in_line[0][0] > -closest_nodes[0][0]:

                        # extracting nearest neighbors from closest_nodes at current layer
                        sorted_closest_nodes = sorted(closest_nodes, key=lambda x: -x[0])
                        for new_neighbor_node_tuple in sorted_closest_nodes:
                            # stop running if reached max neighbors for layer
                            if len(new_node.neighbors_by_layer[cur_layer]) >= (self.max_neighbors0 if cur_layer == 0 else self.max_neighbors):
                                break 
                            # else add neighbor if it passes the select heuristic
                            for already_neighbor_node in new_node.neighbors_by_layer[cur_layer]:
                                if self.get_distance(new_neighbor_node_tuple[1], already_neighbor_node) < -new_neighbor_node_tuple[0]:
                                    break
                            else:
                                new_node.add_neighbor(new_neighbor_node_tuple[1], cur_layer)
                                new_neighbor_node_tuple[1].add_neighbor(new_node, cur_layer)

                        # determining which node to traverse down to next layer
                        next_layer_start_node = sorted_closest_nodes[0][1]
                        next_layer_start_dist = -sorted_closest_nodes[0][0]

                        # clearing visited, next_in_line, and closest_nodes for next layer
                        visited.clear()
                        next_in_line.clear()
                        closest_nodes.clear()

                        # setting up next layer exploration
                        cur_layer -= 1
                        if cur_layer < 0:
                            break
                        visited.add(next_layer_start_node)
                        next_in_line.append((next_layer_start_dist, next_layer_start_node))
                        closest_nodes.append((-next_layer_start_dist, next_layer_start_node))

            # updating entry point if new node has higher level
            if max_level > self.cur_max_level:
                self.entry_point = new_node
                self.cur_max_level = max_level

            return new_node

    def query(self, tensor, k=1):
        if self.entry_point is None:
            return []

        # setup for fasetr search speed by max layer=0
        new_node = vnode(tensor, max_layer=0)

        # db's to track exploration of neighbors (set, min-heap, max-heap)
        visited = set()
        next_in_line = []
        closest_nodes = []

        # start exploration from entry point
        cur_node = self.entry_point
        visited.add(cur_node)
        heapq.heappush(next_in_line, (0, cur_node))

        # exploration loop
        cur_layer = math.inf
        while next_in_line:
            cur_node = heapq.heappop(next_in_line)[1]
            cur_dist = self.get_distance(new_node, cur_node)
            cur_layer = min(cur_layer, cur_node.max_layer)

            # divide exploration into either fast or deep depending on if level is part of new node's levels
            if cur_layer > new_node.max_layer:
                level_min_dist = cur_dist
                level_min_node = cur_node
                changed = False
                # explore only cur_node's neighbors and move down
                for neighbor_node in cur_node.neighbors_by_layer[cur_layer]:
                    if neighbor_node not in visited:
                        visited.add(neighbor_node)
                        neighbor_dist = self.get_distance(new_node, neighbor_node)
                        if neighbor_dist < level_min_dist:
                            level_min_dist = neighbor_dist
                            level_min_node = neighbor_node
                            changed = True

                # traversing down through closest node
                next_in_line.clear()
                heapq.heappush(next_in_line, (level_min_dist, level_min_node))
                
                if not changed:
                    cur_layer -= 1
                    if cur_layer < 0:
                        break
                    # only clear visited and closest_nodes when moving to a new layer!
                    visited.clear()
                    closest_nodes.clear()
                    heapq.heappush(closest_nodes, (-level_min_dist, level_min_node))
            else:
                # if we skipped fast drop, seed the closest_nodes heap
                if len(closest_nodes) == 0:
                    heapq.heappush(closest_nodes, (-cur_dist, cur_node))

                # explore cur_node's neighbors
                for neighbor_node in cur_node.neighbors_by_layer[cur_layer]:
                    if neighbor_node not in visited:
                        visited.add(neighbor_node)
                        neighbor_dist = self.get_distance(new_node, neighbor_node)
                        heapq.heappush(next_in_line, (neighbor_dist, neighbor_node))
                        if len(closest_nodes) < self.ef_construction:
                            heapq.heappush(closest_nodes, (-neighbor_dist, neighbor_node))
                        else:
                            if -neighbor_dist > closest_nodes[0][0]:
                                heapq.heapreplace(closest_nodes, (-neighbor_dist, neighbor_node))

                # determine if we should continue exploring neighbors or move down a layer
                if  len(next_in_line) == 0 or next_in_line[0][0] > -closest_nodes[0][0]:

                    # extracting nearest neighbors from closest_nodes at current layer
                    sorted_closest_nodes = sorted(closest_nodes, key=lambda x: -x[0])

                    # determining which node to traverse down to next layer
                    next_layer_start_node = sorted_closest_nodes[0][1]
                    next_layer_start_dist = -sorted_closest_nodes[0][0]

                    # setting up next layer exploration
                    cur_layer -= 1
                    if cur_layer < 0:
                        return [node_tuple[1] for node_tuple in sorted_closest_nodes[:k]]
                        
                    # clearing visited, next_in_line, and closest_nodes for next layer
                    visited.clear()
                    next_in_line.clear()
                    closest_nodes.clear()

                    visited.add(next_layer_start_node)
                    next_in_line.append((next_layer_start_dist, next_layer_start_node))
                    closest_nodes.append((-next_layer_start_dist, next_layer_start_node))
        
        # emergency return
        return []

    def get_random_level(self):
        r = random.random()
        assigned_level = math.floor(-math.log(r)*self.level_multiplier)
        return max(0, assigned_level)

    def get_distance(self, node1, node2):
        return 1 - torch.dot(node1.normed_tensor, node2.normed_tensor).item()

    # AI CODE -------------------------------------------
    def save(self, filepath="hnsw_database.pt"):
        # 1. Map every node object to its integer index
        node_to_id = {node: i for i, node in enumerate(self.nodes)}

        # 2. Extract tensors into a 2D matrix
        all_tensors = torch.stack([node.tensor for node in self.nodes])

        # 3. Convert cyclic graph to integers
        graph_structure = []
        for node in self.nodes:
            node_graph = []
            for layer in node.neighbors_by_layer:
                neighbor_ids = [node_to_id[neighbor] for neighbor in layer]
                node_graph.append(neighbor_ids)
            graph_structure.append(node_graph)

        # 4. Save metadata, including the two new attributes
        state_dict = {
            "max_neighbors": self.max_neighbors,
            "ef_construction": self.ef_construction,
            "cur_max_level": self.cur_max_level,
            "entry_point_id": node_to_id[self.entry_point] if self.entry_point else None,
            "node_max_layers": [node.max_layer for node in self.nodes],
            "tensors": all_tensors,
            "graph_structure": graph_structure,
            "node_keys": [node.gaph_db_key for node in self.nodes] 
        }

        torch.save(state_dict, filepath)
        print(f"Successfully saved {len(self.nodes)} nodes to {filepath}")

    @classmethod
    def load(cls, filepath="hnsw_database.pt"):
        state_dict = torch.load(filepath)

        # 1. Re-initialize database
        db = cls(
            max_neighbors=state_dict["max_neighbors"], 
            ef_construction=state_dict["ef_construction"]
        )
        db.cur_max_level = state_dict["cur_max_level"]

        tensors = state_dict["tensors"]
        node_max_layers = state_dict["node_max_layers"]
        graph_structure = state_dict["graph_structure"]
        
        # Safely get the text and keys (defaults to None if loading an old save)
        node_keys = state_dict.get("node_keys", [None] * len(tensors))

        # 2. Re-create all bare nodes and attach the new attributes
        for i in range(len(tensors)):
            new_node = vnode(
                tensor=tensors[i], 
                max_layer=node_max_layers[i], 
                max_neighbors=state_dict["max_neighbors"]
            )
            # Assign the saved text and key
            new_node.gaph_db_key = node_keys[i]
            
            db.nodes.append(new_node)

        # 3. Rewire edges
        for i, node in enumerate(db.nodes):
            for layer, neighbor_ids in enumerate(graph_structure[i]):
                for n_id in neighbor_ids:
                    node.neighbors_by_layer[layer].append(db.nodes[n_id])

        # 4. Restore entry point
        entry_id = state_dict["entry_point_id"]
        if entry_id is not None:
            db.entry_point = db.nodes[entry_id]

        print(f"Successfully loaded {len(db.nodes)} nodes from {filepath}")
        return db
    # AI CODE -------------------------------------------

    
class vnode:
    def __init__(self, tensor=None, max_layer=0, max_neighbors=16):
        self.tensor = tensor
        self.max_layer = max_layer
        self.max_layer_neighbors = max_neighbors
        self.normed_tensor = F.normalize(tensor, p=2, dim=0) if tensor is not None else None
        self.neighbors_by_layer = [[] for _ in range(max_layer + 1)]
        self.graph_db_key = None

    def add_neighbor(self, neighbor_node, layer):
        if (layer == 0 and len(self.neighbors_by_layer[layer]) >= self.max_layer_neighbors*2) \
                or (layer > 0 and len(self.neighbors_by_layer[layer]) >= self.max_layer_neighbors):
            # building temp list with tuples of (distance, neighbor_node) & sort
            self.neighbors_by_layer[layer].append(neighbor_node)
            temp_tuple_neighbor_list = []

            for neighbor_node in self.neighbors_by_layer[layer]:
                temp_tuple_neighbor_list.append((self.get_distance(neighbor_node), neighbor_node))

            temp_tuple_neighbor_list.sort(key=lambda x: x[0])

            # use select heuristic to choose closest neighbors
            new_neighbors_list = []
            for neighbor_tuple in temp_tuple_neighbor_list:
                if len(new_neighbors_list) == 0:
                    new_neighbors_list.append(neighbor_tuple[1])
                else:
                    for already_neighbor_node in new_neighbors_list:
                        if neighbor_tuple[1].get_distance(already_neighbor_node) < neighbor_tuple[0]:
                            break
                    else:
                        new_neighbors_list.append(neighbor_tuple[1])

            self.neighbors_by_layer[layer] = new_neighbors_list
        else:
            self.neighbors_by_layer[layer].append(neighbor_node)

    def get_distance(self, other_node):
        return 1 - torch.dot(self.normed_tensor, other_node.normed_tensor).item()

