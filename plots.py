import json
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os

def create_folder_if_not_exists(folder_path):
    """Create folder if it does not exist.
    
    Args:
        folder_path (str): Path to folder.
    
    Returns:
        str: Message about whether folder was created or not.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return f"Folder '{folder_path}' created."
    else:
        return f"Folder '{folder_path}' already exists."

def visualize_train_val_dynamics(train_result_path='experiments/reproduction/outputs/liverfailure/logs/train_log.json', 
                                 val_result_path='experiments/reproduction/outputs/liverfailure/logs/val_log.json', 
                                 out_dir="Train_val_dynamics.png", 
                                 out_file_name="Train_val_dynamics.png",
                                 filetypes = ["png", "pdf"],
                                 steps_per_epoch=None):
    with open(train_result_path, 'r') as f:
        train_res = json.load(f)
    
    with open(val_result_path, 'r') as f:
        val_res = json.load(f)
    
    n_epochs = len(train_res['acc'])-1
    epochs = list(range(1, n_epochs + 1))

    if steps_per_epoch:
        plt.xlabel('Steps')
        epochs = [ep*steps_per_epoch for ep in epochs]
    else:
        plt.xlabel('Epochs')

    train_losses = [np.mean(e) for e in train_res['loss']]
    val_losses = [np.mean(e) for e in val_res['loss']]

    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='validation loss')
    plt.plot(epochs, train_res['acc'], label='Training accuracy')
    plt.plot(epochs, val_res['acc'], label='validation accuracy')
    
    plt.title('Training and Validation loss & acc')
    
    #plt.ylabel('Loss')
    plt.legend()

    out_dir_split = out_dir.split('/')
    print(create_folder_if_not_exists(out_dir))

    # split out_file_name if it contains .
    out_file_name_split = out_file_name.split('.')
    out_file_name_split = out_file_name_split[0]

    # add filetypes to out_file_name_split
    for filetype in filetypes:
        out_file_name_split
    
        # join out_file_name_split
        out_file_name = f'{out_file_name_split}.{filetype}'

        out_file_path = os.path.join(*out_dir_split, out_file_name)

        print(f"Saving plot to {out_file_path}")
        plt.savefig(out_file_path)
    plt.clf()

if __name__ == '__main__':
    visualize_train_val_dynamics(train_result_path='experiments/reproduction/outputs/liverfailure/logs/train_log.json',
                                val_result_path='experiments/reproduction/outputs/liverfailure/logs/val_log.json',
                                out_dir="plots/liverfailure", 
                                out_file_name="train_validation_loss_acc.",
                                steps_per_epoch=184)
    visualize_train_val_dynamics(train_result_path='experiments/reproduction/outputs/tramadol/logs/train_log.json',
                                val_result_path='experiments/reproduction/outputs/tramadol/logs/val_log.json',
                                out_dir="plots/tramadol", 
                                out_file_name="train_validation_loss_acc",
                                steps_per_epoch=128)