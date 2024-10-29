import copy
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm


def evaluate_metrics(model, test_loader):
    model.eval()
    with torch.no_grad():
        acc = 0
        for data in tqdm(test_loader):
            out = model(data[:-1])
            temp_acc = 0
            for i in range(len(out)):
                loop_set = loop_calculation(out[i], data[-1][i])
                max_acc = -999
                for pos_ in loop_set:
                    tmp_acc = accuracy_score(pos_, data[-1][i])
                    if tmp_acc > max_acc:
                        max_acc = tmp_acc
                temp_acc += max_acc
            temp_acc = temp_acc / len(out)
            acc += temp_acc
        acc = acc / len(test_loader)
        print("Average Accuracy: ", acc)


def loop_calculation(input_1, input_2):
    out_ = []
    input_set = set(input_1)
    label_set = set(input_2)
    pairs = loop_check(label_set, input_set)
    for pair in pairs:
        tem_input = copy.deepcopy(input_1)
        changed = np.zeros(len(tem_input))
        for pair_info in pair:
            original_label = pair_info[0]
            replace_label = pair_info[1]
            for i in range(len(tem_input)):
                if tem_input[i] == original_label and changed[i] == 0:
                    tem_input[i] = replace_label
                    changed[i] = 1
        for i in range(len(changed)):
            if changed[i] == 0:
                tem_input[i] = 0
        out_.append(tem_input)
    return out_


def loop_check(label_set, input_set):
    set_pairs = []
    for label in label_set:
        for input_label in input_set:
            if len(label_set) > 1 and len(input_set) > 1:
                a_ = copy.deepcopy(label_set)
                a_.remove(label)
                b_ = copy.deepcopy(input_set)
                b_.remove(input_label)
                get_pairs = loop_check(a_, b_)
                for pair in get_pairs:
                    tmp = pair
                    tmp.append([input_label, label])
                    set_pairs.append(tmp)
            elif len(label_set) == 1 and len(input_set) == 1:
                return [[[input_label, label]]]
            else:
                set_pairs.append([[input_label, label]])
    for i in range(len(set_pairs)):
        set_pairs[i].sort()
    temp = []
    for item in set_pairs:
        if item not in temp:
            temp.append(item)
    return temp


def data_reformat(input_data, label):
    max_ = 0
    for label_ in label:
        if label_ > max_:
            max_ = label_
    max_ = max_ + 1
    output_d = []
    for data_ in input_data:
        new_data = []
        for i in range(max_):
            if data_ == i + 1:
                new_data.append(1)
            else:
                new_data.append(0)
        output_d.append(new_data)
    return output_d


def train(epochs, trainLoader, testLoader, model, learning_rate, model_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    try:
        for e in range(epochs):
            for index, _data in enumerate(tqdm(trainLoader, leave=False)):
                model.train()
                out = model(_data[:-1])
                y_ = _data[-1]
                total_loss = 0
                for i in range(len(out)):
                    loop_set = loop_calculation(out[i], y_[i])
                    min_loss = float('inf')
                    for data_setting in loop_set:
                        temp_loss = criterion(
                            torch.tensor(data_reformat(data_setting, y_[i]), dtype=torch.float),
                            torch.tensor(y_[i])
                        )
                        if temp_loss < min_loss:
                            min_loss = temp_loss
                    total_loss += min_loss
                loss = torch.autograd.Variable(total_loss, requires_grad=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 20 == 0:
                    print(f'epoch: {e + 1}, batch: {index + 1}, loss: {loss.data}')

            if model_path:
                torch.save(model, model_path)
                print(f"Model saved at {model_path}")

        evaluate_metrics(model=model, test_loader=testLoader)
    
    except KeyboardInterrupt:
        evaluate_metrics(model=model, test_loader=testLoader)

