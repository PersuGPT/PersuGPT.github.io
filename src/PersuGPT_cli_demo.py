from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")


def main():
    chat_model = ChatModel()

    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
    task_prompt_template = "You are a professional persuader. Your identity: {}\nThe persuasion scenario: {}\nThe persuasion task: {}. \nNext, you need first to give candidate persuasion strategies, then analyze, select strategy, and interaction."
    
    while True:
        history = []
        persuader = input("persuader:")
        context = input("context:")
        task = input("task:")
        user_input = task_prompt_template.format(persuader, context, task)

        first_turn_falg = True
        while True:
            if first_turn_falg:
                first_turn_falg = False
                query = user_input
                print(query)
            else:
                try:
                    query = input("\nUser: ")
                except UnicodeDecodeError:
                    print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
                    continue
                except Exception:
                    raise
            if query.strip() == "exit":
                break
            if query.strip() == "clear":
                history = []
                torch_gc()
                print("History has been removed.")
                continue

            print("Assistant: ", end="", flush=True)

            response = ""
            for new_text in chat_model.stream_chat(query, history):
                print(new_text, end="", flush=True)
                response += new_text
            print()

            history = history + [(query, response)]


if __name__ == "__main__":
    main()
