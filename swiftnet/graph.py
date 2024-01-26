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

def get_final_epoch_stats(model_name, data_path, natural=False):
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
    
    print(f'Final epoch stats for model {model_name}')
    
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

        final_epoch_val = data[-1]
        print(f'{name}: {final_epoch_val}')

    print()

if __name__ == '__main__':
    # graph_stats(model_name='SwiftNet, Natural data', data_path='./results/23-57_rn18_single_scale/', natural=True, store_path='./graphs/natural')

    # graph_stats(model_name='SwiftNet, Non-semantic poisoning', data_path='./results/31-95_rn18_single_scale_ns/', natural=False, store_path='./graphs/nonsemantic')
    # graph_stats(model_name='SwiftNet, Non-semantic poisoning (Frame)', data_path='./results/32-05_rn18_single_scale_frame_ns/', natural=False, store_path='./graphs/nonsemantic_frame')
    # graph_stats(model_name='SwiftNet, Non-semantic poisoning (Frame, 50% Poisoning)', data_path='./results/26-95_rn18_single_scale_frame_ns/', natural=False, store_path='./graphs/nonsemantic_frame_0.5')

    # graph_stats(model_name='SwiftNet, Fine-grained non-semantic poisoning', data_path='./results/32-50_rn18_single_scale_ns_fg/', natural=False, store_path='./graphs/nonsemantic_finegrained')
    # graph_stats(model_name='SwiftNet, Semantic poisoning', data_path='./results/29-30_rn18_single_scale_s/', natural=False, store_path='./graphs/semantic')
    # graph_stats(model_name='SwiftNet, Fine-grained semantic poisoning', data_path='./results/33-07_rn18_single_scale_s_fg/', natural=False, store_path='./graphs/semantic_finegrained')

    get_final_epoch_stats(model_name='SwiftNet, Natural data', data_path='./results/33-36_rn18_single_scale/', natural=True)

    get_final_epoch_stats(model_name='SwiftNet, Non-semantic poisoning', data_path='./results/31-95_rn18_single_scale_ns/', natural=False)
    get_final_epoch_stats(model_name='SwiftNet, Non-semantic poisoning (Frame)', data_path='./results/32-05_rn18_single_scale_frame_ns/', natural=False)
    get_final_epoch_stats(model_name='SwiftNet, Non-semantic poisoning (Frame, 50% Poisoning)', data_path='./results/26-95_rn18_single_scale_frame_ns/', natural=False)

    get_final_epoch_stats(model_name='SwiftNet, Fine-grained non-semantic poisoning', data_path='./results/32-50_rn18_single_scale_ns_fg/', natural=False)
    get_final_epoch_stats(model_name='SwiftNet, Semantic poisoning', data_path='./results/29-30_rn18_single_scale_s/', natural=False)
    get_final_epoch_stats(model_name='SwiftNet, Fine-grained semantic poisoning', data_path='./results/33-07_rn18_single_scale_s_fg/', natural=False)