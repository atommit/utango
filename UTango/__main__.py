import sys
import os
from .run import demo_work, start_training

DATA_DIR = "data"
MODELS_DIR = "models"

sys.path.append(os.path.dirname(__file__))

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m UTango <action> [<model_path>] [<dataset_folder>] [<num_data_files>]")
        sys.exit(1)

    action = sys.argv[1]
    
    model_path = os.path.join(MODELS_DIR, "demo_model.pt")
    dataset_folder = os.path.join(DATA_DIR, "demo")
    num_files = 5

    if action == "demo":
        if len(sys.argv) > 2:
            model_path = os.path.join(MODELS_DIR, sys.argv[3])
        if len(sys.argv) > 3:
            dataset_folder = os.path.join(DATA_DIR, sys.argv[2])
        if len(sys.argv) > 4:
            num_files = int(sys.argv[4])

        demo_work(model_path, dataset_folder, num_files)

    elif action == "train":
        if len(sys.argv) < 4:
            print("Usage for training: python -m UTango train <model_path> <dataset_folder> <num_data_files>")
            sys.exit(1)

        model_path = os.path.join(MODELS_DIR, sys.argv[2])
        dataset_folder = os.path.join(DATA_DIR, sys.argv[3])
        num_files = int(sys.argv[4])

        start_training(model_path, dataset_folder, num_files)

    else:
        print("Unknown action:", action)
        sys.exit(1)

if __name__ == '__main__':
    main()
