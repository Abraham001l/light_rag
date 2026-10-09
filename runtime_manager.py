import graph_db
import model
import vector_db
import time

class runtime_manager:
    def __init__(self):
        self.graph_db = None
        self.vector_db = None
        self.model = model.model()

    def run(self):
        print("Hi, I am your runtime manager. I will help you manage your graph and model.")
        print("\nFirst I would like to ask you, do you have a saved graph which you would like to load? (y/n)")
        user_input = input().lower()
        if user_input == "y":
            print("Please enter the path to your save vector database file:")
            # file_path = input()
            self.vector_db = vector_db.hnsw_database.load("hnsw_database.pt")
            print("Vector database loaded successfully.")
            print("Please enter the path to your save graph database file:")
            # file_path = input()
            self.graph_db = graph_db.graph.load("semantic_graph.pkl")
            print("Graph database loaded successfully.")
        else:
            print("Okay, we will start with a new graph and vector database.")
            self.vector_db = vector_db.hnsw_database()
            self.graph_db = graph_db.graph()

        while True:
            print("\nOkay great let's get to work.\nEnter the following command to add to the database: 'add <text>'\nEnter the following command to ask a question: 'ask <text>'\nEnter 'exit' to exit the program.")

            # grab user input
            user_input = input()

            # handling add case
            if user_input.startswith("add "):
                text_to_add = user_input[4:]

                # prompt the model to generate entities and relationships
                prompt = self.define_entity_gen_prompt(text_to_add)
                response = self.model.generate_text([prompt])[0]
                entities, relationships = self.parse_model_response(response)

                # adding entities to graph and vector database
                for entity_name, entity_type, entity_description in entities:
                    # adding to graph database
                    description = f"Name: {entity_name}\nType: {entity_type}\nDescription: {entity_description}"
                    added_new_gnode = self.graph_db.add_node(entity_name, description)

                    # test print
                    print(f"Added entity: {entity_name}, Type: {entity_type}, Description: {entity_description}")

                    # adding to vector database
                    if added_new_gnode:
                        embedding = self.model.get_embeddings([description])[0]
                        new_vnode = self.vector_db.add_node(embedding)
                        # associate the new vector node with the graph node
                        new_vnode.gaph_db_key = entity_name

                # adding relationships to graph database and vector database
                for source_entity, target_entity, relationship_description, relationship_strength in relationships:
                    # adding to graph database
                    self.graph_db.add_neighbors_to_node(source_entity, [target_entity])

            elif user_input.startswith("ask "):
                question = user_input[4:]

                # get embedding and query vector db
                question_embedding = self.model.get_embeddings([question])[0]
                most_similar_node = self.vector_db.query(question_embedding, k=1)[0]

                # retrieve the corresponding graph node
                graph_node_key = most_similar_node.gaph_db_key
                graph_node = self.graph_db.get_node(graph_node_key)

                if graph_node:
                    print(f"Most relevant information found:\n{graph_node.description}")

                    # print the neighbors of the graph node
                    neighbors = self.graph_db.get_neighbors_of_node(graph_node_key)
                    if neighbors:
                        print("\nNeighbors of the most relevant information:")
                        for neighbor_key in neighbors:
                            neighbor_node = self.graph_db.get_node(neighbor_key)
                            if neighbor_node:
                                print(f"- {neighbor_node.description}")
                    else:
                        print("No neighbors found for the most relevant information.")
            else:
                # exit loop to save and exit
                break

        # save the graph and vector database before exiting
        print("Saving the graph and vector database before exiting...")
        self.vector_db.save("hnsw_database.pt")
        self.graph_db.save("semantic_graph.pkl")

    def parse_model_response(self, response):
        # split the response into records
        records = response.split("##")

        # initialize empty lists for entities and relationships
        entities = []
        relationships = []

        # iterate through each record
        for record in records:
            # check if the record is an entity or a relationship
            if record.startswith('("entity"'):
                # parse the entity record
                parts = record.split("<|>")
                entity_name = parts[1].strip()
                entity_type = parts[2].strip()
                entity_description = parts[3].strip()
                entities.append((entity_name, entity_type, entity_description))
            elif record.startswith('("relationship"'):
                # parse the relationship record
                parts = record.split("<|>")
                source_entity = parts[1].strip()
                target_entity = parts[2].strip()
                relationship_description = parts[3].strip()
                relationship_strength = float(parts[4].strip())
                relationships.append((source_entity, target_entity, relationship_description, relationship_strength))

        return entities, relationships

    def define_entity_gen_prompt(self, input_text, tuple_delimiter="<|>", record_delimiter="##", entity_types=["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"], completion_delimiter="<|COMPLETE|>"):
        return (f"""---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify
all entities of those types from the text and all relationships among the identified entities.
---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity name: Name of the entity, capitalized
- entity type: One of the following types: [{entity_types}]
- entity description: Comprehensive description of the entity’s attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity name>{tuple_delimiter}<entity type>{tuple_delimiter}<entity description>
2. From the entities identified in step 1, identify all pairs of (source entity, target entity) that
are *clearly related* to each other
For each pair of related entities, extract the following information:
- source entity: name of the source entity, as identified in step 1
- target entity: name of the target entity, as identified in step 1
- relationship description: explanation as to why you think the source entity and the target entity are
related to each other
- relationship strength: a numeric score indicating strength of the relationship between the source entity
and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source entity>{tuple_delimiter}<target
entity>{tuple_delimiter}<relationship description>{tuple_delimiter}<relationship strength>)
3. Return output in English as a single list of all the entities and relationships identified in steps 1
and 2. Use **{record_delimiter}** as the list delimiter.
4. When finished, output {completion_delimiter}
---Examples---
Entity types: ORGANIZATION,PERSON
Input:
The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank planning to release its
latest policy decision on Wednesday at 2:00 p.m. ET, followed by a press conference where Fed Chair
Jerome Powell will take questions. Investors expect the Federal Open Market Committee to hold its
benchmark interest rate steady in a range of 5.25%-5.5%.
Output:
("entity"{tuple_delimiter}FED{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Fed is the Federal Reserve,
which is setting interest rates on Tuesday and Wednesday)
{record_delimiter}
("entity"{tuple_delimiter}JEROME POWELL{tuple_delimiter}PERSON{tuple_delimiter}Jerome Powell is the chair
of the Federal Reserve)
{record_delimiter}
("entity"{tuple_delimiter}FEDERAL OPEN MARKET COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The
Federal Reserve committee makes key decisions about interest rates and the growth of the United States
money supply)
{record_delimiter}
("relationship"{tuple_delimiter}JEROME POWELL{tuple_delimiter}FED{tuple_delimiter}Jerome Powell is the
Chair of the Federal Reserve and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}
...More examples...
---Real Data---
Entity types: {entity_types}
Input:
{input_text}
Output:
""")
            

r = runtime_manager()
r.run()