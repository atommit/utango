import os
import random
import torch
from . import model
from .train import train, evaluate_metrics

TRAIN_PERCENTAGE = 0.8

def demo_work(model_path, dataset_folder, num_files):
    dataset = load_dataset(dataset_folder, num_files)
    model_ = torch.load(model_path)

    evaluate_metrics(model=model_, test_loader=dataset)
    print("Demo results displayed.")

def load_dataset(folder, num_files):
    dataset = []
    root = os.getcwd()
    for i in range(num_files):
        data = torch.load(os.path.join(root, folder, f'data_{i}.pt'))
        dataset.append(data)
    return dataset

def split_dataset(dataset):
    random.shuffle(dataset)

    train_num = int(len(dataset) * TRAIN_PERCENTAGE)
    train_dataset = dataset[:train_num]
    test_dataset = dataset[train_num:]

    return train_dataset, test_dataset

def start_training(model_path, dataset_folder, num_files):
    dataset = load_dataset(dataset_folder, num_files)
    train_dataset, test_dataset = split_dataset(dataset)
    
    if model_path and os.path.exists(model_path):
        model_ = torch.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        model_ = model.UTango(h_size=128, max_context=5, drop_out_rate=0.5, gcn_layers=3)
        print("Initialized a new model.")

    train(epochs=1, trainLoader=train_dataset, testLoader=test_dataset, model=model_, learning_rate=0.0001, model_path=model_path)

