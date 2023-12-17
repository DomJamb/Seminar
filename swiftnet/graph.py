import matplotlib.pyplot as plt
import pickle
import os

def graph_miou(model_name, data_path, store_path=None):
    file = 'val_ious.pkl'
    path = data_path + file
    with open(path, 'rb') as file:
        data = pickle.load(file)

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(data) + 1), data, label=model_name)
    plt.title('MIoU over epochs', fontsize=20)
    plt.xlabel('Epoch number', fontsize=16)
    plt.ylabel('MIoU', fontsize=16)
    plt.legend(loc='upper left')

    if store_path:
        if not os.path.exists(store_path): 
            os.makedirs(store_path)

        plt.savefig(f'{store_path}/miou.png')
    else:
        plt.show()

def graph_stats(model_name, data_path, store_path=None):
    files = {'Validation MIoU': 'val_ious.pkl', 
             'Validation PA': 'val_pas.pkl',
             'Poisoned Validation MIoU': 'val_poisoned_ious.pkl',
             'Poisoned Validation ASR': 'val_asrs.txt'
             }
    
    for i, (name, file) in enumerate(files.items()):
        path = data_path + file
        if path.endswith(".txt"):
            with open(path, 'r') as f:
                raw_data = f.readlines()
                data = [float(x.split(': ')[1][:-2]) for x in raw_data]
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)

        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(data) + 1), data)
        plt.title(f'{model_name}, {name}', fontsize=20)
        plt.xlabel('Epoch number', fontsize=16)
        plt.ylabel(name.split(" ")[-1], fontsize=16)

        if store_path:
            if not os.path.exists(store_path): 
                os.makedirs(store_path)

            plt.savefig(f'{store_path}/{name.replace(" ", "_").lower()}.png')
        else:
            plt.show()

if __name__ == '__main__':
    # graph_miou('SwiftNet, Natural data', './results/33-36_rn18_single_scale/', './graphs/natural')
    graph_stats('SwiftNet, Non-semantic poisoning', './results/22-70_rn18_single_scale_ns/', './graphs/nonsemantic')