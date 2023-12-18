import matplotlib.pyplot as plt
import pickle
import os

def graph_stats(model_name, data_path, natural=False, store_path=None):
    files = {
        'Validation MIoU': 'val_ious.pkl', 
        'Validation PA': 'val_pas.pkl',
    }
    
    if not natural:
        poisoned_files = {
            'Poisoned Validation MIoU': 'val_poisoned_ious.pkl',
            'Poisoned Validation ASR': 'val_asrs.txt'
        }
        
        files = {**files, **poisoned_files}
    
    for i, (name, file) in enumerate(files.items()):
        path = data_path + file
        if not os.path.isfile(path):
            continue

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
    graph_stats(model_name='SwiftNet, Natural data', data_path='./results/23-57_rn18_single_scale/', natural=True, store_path='./graphs/natural')
    # graph_stats(model_name='SwiftNet, Non-semantic poisoning', data_path='./results/22-70_rn18_single_scale_ns/', natural=False, store_path='./graphs/nonsemantic')