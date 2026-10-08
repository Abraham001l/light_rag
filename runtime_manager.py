import graph_db
import model

class runtime_manager:
    def __init__(self):
        self.graph = graph_db.graph()
        self.model = model.model()

    def run(self):
        print("Hi, I am your runtime manager. I will help you manage your graph and model.")
        print("\nFirst I would like to ask you, do you have a saved graph which you would like to load? (y/n)")
        user_input = input().lower()
        if user_input == "y":
            pass

        print("\nOkay great let's get to work.\nEnter the following command to add to the database: 'add <text>'\nEnter the following command to ask a question: 'ask <text>'")

r = runtime_manager()
r.run()