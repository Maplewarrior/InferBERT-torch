import json
import matplotlib.pyplot as plt
import numpy as np
import pdb

def visualize_train_val_dynamics(train_result_path='experiments/reproduction/outputs/liverfailure/logs/train_log.json', val_result_path='experiments/reproduction/outputs/liverfailure/logs/val_log.json'):
    with open(train_result_path, 'r') as f:
        train_res = json.load(f)
    
    with open(val_result_path, 'r') as f:
        val_res = json.load(f)
    
    n_epochs = len(train_res['acc'])-1
    epochs = list(range(1, n_epochs + 1))
    train_losses = [np.mean(e) for e in train_res['loss']]
    val_losses = [np.mean(e) for e in val_res['loss']]

    plt.plot(epochs, train_losses[:-1], 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='validation loss')
    plt.plot(epochs, train_res['acc'][:-1], 'y', label='Training accuracy')
    plt.plot(epochs, val_res['acc'], 'r', label='validation accuracy')
    
    plt.title('Training and Validation loss & acc')
    plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Train_val_dynamics.png")

if __name__ == '__main__':
    visualize_train_val_dynamics()