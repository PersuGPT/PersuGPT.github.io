import json
import random
import jsonlines


test_file_path = "/Users/chuhao/my_project/PersuasiveLLM/backup_from_bingxing/data_collect/en_data_final/test_1000.json"

f = open(test_file_path, "r")
test_data = json.load(f)
f.close()

# print(test_data[0])

example_list = []
for _ in range(100):
    rand_id = random.randint(0, len(test_data) - 1)
    example = test_data[rand_id]

    # print("\n\n============== Task Description ===============\n\n")
    domain = example["task"]["theme"][0]["theme"][0]
    background = example["task"]["context"]
    goal = example["task"]["goal"]

    # print("background:", background)
    # print("goal:", goal)

    # print("\n\n============== Strategy ===============\n\n")
    strategy = "; ".join(example["strategy"])
    # print(strategy)

    task_info = f"<b>Background</b>: {background}</br></br><b>Goal</b>: {goal}</br></br><b>Candidate Strategies</b>: {strategy}"
    # print(task_info)

    # print("\n\n============== Dialogue ===============\n\n")
    dial_list = []
    dialogue = example["dialog"][0]["robot_resp"]
    for i, item in enumerate(dialogue):
        role = item["role"]
        response = item["response"]
        # if role == "persuadee":
            # print(f"{role}: {response}\n=========\n")
            # dial_list.append(f"{response}")
        if role == "persuader":
            analysis = item["analysis"]
            selected_strategy = item["strategy"]
            # print(f"{role}: Intent to Strategy Reasoning: {analysis}\nSelected Strategy: {selected_strategy}\nResponse: {response}")
            dial_list.append(f"<b>Intent to Strategy Reasoning</b>: {analysis}</br><b>Selected Strategy</b>: {selected_strategy}")
        dial_list.append(f"{response}")

    # print(dial_list)
    # print(dialogue)
    example_list.append((domain, [task_info] + dial_list))

with jsonlines.open("example.jsonl", mode = "w") as writer:
    for obj in example_list:
        writer.write(obj)

