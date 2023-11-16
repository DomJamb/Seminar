import matplotlib.pyplot as plt
import pickle

def graph_miou(data, model_name, store_path=None):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(data) + 1), data, label=model_name)
    plt.title('MIoU over epochs', fontsize=20)
    plt.xlabel('Epoch number', fontsize=16)
    plt.ylabel('MIoU', fontsize=16)
    plt.legend(loc='upper left')

    if store_path:
        plt.savefig(store_path)
    else:
        plt.show()

if __name__ == '__main__':
    base_path = './results'
    file = 'val_ious.pkl'
    path = base_path + '/33-36_rn18_single_scale/' + file
    with open(path, 'rb') as file:
        miou = pickle.load(file)
    
    graph_miou(miou, 'SwiftNet, Natural data', './graphs/natural/miou.png')