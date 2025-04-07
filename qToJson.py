import torch
import json

def save_q_table_as_json(filename="q_learning_model.pt", output_file="q_learning_model.json"):
    try:
        Q_dict = torch.load(filename)
        # Convert keys to strings for JSON compatibility
        Q_dict_readable = {str(k): v.tolist() for k, v in Q_dict.items()}
        with open(output_file, "w") as f:
            json.dump(Q_dict_readable, f, indent=4)
        print(f"Q-table saved as {output_file}")
    except Exception as e:
        print(f"Error converting the file: {e}")

if __name__ == "__main__":
    save_q_table_as_json()