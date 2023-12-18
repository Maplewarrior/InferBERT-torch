import json
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
import matplotlib.colors as mcolors

def get_losses_acc(train_result_path, val_result_path):
    with open(train_result_path, 'r') as f:
        train_res = json.load(f)
    
    with open(val_result_path, 'r') as f:
        val_res = json.load(f)
    
    n_epochs = len(train_res['acc'])
    epochs = list(range(1, n_epochs + 1))

    train_losses = [np.mean(e) for e in train_res['loss']]
    val_losses = [np.mean(e) for e in val_res['loss']]

    return train_losses, val_losses, train_res['acc'], val_res['acc'], epochs

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
                                 steps_per_epoch=None,
                                 plot_title = "Training and Validation loss & acc"):
    with open(train_result_path, 'r') as f:
        train_res = json.load(f)
    
    with open(val_result_path, 'r') as f:
        val_res = json.load(f)
    
    n_epochs = len(train_res['acc'])
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
    
    plt.title(plot_title)
    
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

def plot_side_by_side(exp_folder_path_1, exp_folder_path_2, out_file_name='plots/train_val_loss_acc.pdf'):

    exp_dir_liver = exp_folder_path_1
    exp_dir_tramadol = exp_folder_path_2    
    # exp_dir_liver = "experiments/reproduction/outputs/liverfailure"
    # exp_dir_tramadol = "experiments/reproduction/outputs/tramadol"
    # exp_dir_tramadol_corr = "experiments/reproduction/outputs/tramadol_corrected"

    exp_num = 1 

    train_res_liver = f'{exp_dir_liver}_{exp_num}/logs/train_log.json'
    test_res_liver = f'{exp_dir_liver}_{exp_num}/logs/test_log.json'
    train_res_tramadol = f'{exp_dir_tramadol}_{exp_num}/logs/train_log.json'
    test_res_tramadol = f'{exp_dir_tramadol}_{exp_num}/logs/test_log.json'
    # train_res_tramadol_corr = f'{exp_dir_tramadol_corr}_{exp_num}/logs/train_log.json'
    # test_res_tramadol_corr = f'{exp_dir_tramadol_corr}_{exp_num}/logs/test_log.json'

    train_loss_liver, test_loss_liver, train_acc_liver, test_acc_liver, epochs_liver = get_losses_acc(train_res_liver, test_res_liver)
    train_loss_tramadol, test_loss_tramadol, train_acc_tramadol, test_acc_tramadol, epochs_tramadol = get_losses_acc(train_res_tramadol, test_res_tramadol)
    # train_loss_tramadol_corr, test_loss_tramadol_corr, train_acc_tramadol_corr, test_acc_tramadol_corr, epochs_tramadol_corr = get_losses_acc(train_res_tramadol_corr, test_res_tramadol_corr)

    steps_per_epoch_liver = 184
    steps_per_epoch_tramadol = 128
    
        
    # Load the original "Blues" colormap
    cmap_greens = plt.cm.Greens
    cmap_reds = plt.cm.Reds

    # Create a custom colormap that excludes the lighter part
    # For example, use only the colors from 0.3 to 1.0 of the original colormap
    start = 0.3
    stop = 1.0
    colors_greens = cmap_greens(np.linspace(start, stop, cmap_greens.N))
    custom_cmap_greens = mcolors.LinearSegmentedColormap.from_list('custom_greens', colors_greens)
    colors_reds = cmap_reds(np.linspace(start, stop, cmap_reds.N))
    custom_cmap_reds = mcolors.LinearSegmentedColormap.from_list('custom_reds', colors_reds)


    # plot 3 figures side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))


    ax = axs[0]
    ax.set_title("Analgesic-induced Liver Failure", fontsize=10)
    if steps_per_epoch_liver:
        ax.set_xlabel('Steps')
        epochs_liver = [ep*steps_per_epoch_liver for ep in epochs_liver]
    else:
        plt.set_xlabel('Epochs')
    # plot liverfailure
    ax.plot(epochs_liver, train_loss_liver, label='Training loss', color=custom_cmap_reds(0.5), linestyle='--')
    ax.plot(epochs_liver, test_loss_liver, label='Validation loss', color=custom_cmap_reds(0.5))
    ax.plot(epochs_liver, train_acc_liver, label='Training accuracy', color = custom_cmap_greens(0.5), linestyle='--')
    ax.plot(epochs_liver, test_acc_liver, label='Validation accuracy', color = custom_cmap_greens(0.5))
    # set y between 0 and 1
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    ax = axs[1]
    ax.set_title("Tramadol-related mortalities", fontsize=10)
    if steps_per_epoch_tramadol:
        ax.set_xlabel('Steps')
        epochs_tramadol = [ep*steps_per_epoch_tramadol for ep in epochs_tramadol]
    else:
        plt.set_xlabel('Epochs')
    # plot tramadol
    ax.plot(epochs_tramadol, train_loss_tramadol, label='Training loss', color=custom_cmap_reds(0.5), linestyle='--')
    ax.plot(epochs_tramadol, test_loss_tramadol, label='Validation loss', color=custom_cmap_reds(0.5))
    ax.plot(epochs_tramadol, train_acc_tramadol, label='Training accuracy', color = custom_cmap_greens(0.5), linestyle='--')
    ax.plot(epochs_tramadol, test_acc_tramadol, label='Validation accuracy', color = custom_cmap_greens(0.5))
    # ax.plot(epochs_tramadol, train_acc_tramadol_corr, label='Training accuracy (corrected)', color = custom_cmap_greens(0.5), linestyle='--')
    # ax.plot(epochs_tramadol, test_acc_tramadol_corr, label='Validation accuracy (corrected)', color = custom_cmap_greens(0.5))
    ax.set_ylim(0, 1)


    # ax = axs[2]
    # ax.set_title("Tramadol-related mortalities (corrected)")
    # if steps_per_epoch_tramadol:
    #     ax.set_xlabel('Steps')
    #     epochs_tramadol_corr = [ep*steps_per_epoch_tramadol for ep in epochs_tramadol_corr]
    # else:
    #     plt.set_xlabel('Epochs')
    # # plot tramadol corrected
    # ax.plot(epochs_tramadol_corr, train_loss_tramadol_corr, label='Training loss')
    # ax.plot(epochs_tramadol_corr, test_loss_tramadol_corr, label='Validation loss')

    # ax.plot(epochs_tramadol_corr, train_acc_tramadol_corr, label='Training accuracy')
    # ax.plot(epochs_tramadol_corr, test_acc_tramadol_corr, label='Validation accuracy')
    # ax.set_ylim(0, 1)

    

    fig.suptitle("Training and validation loss & accuracy", fontsize=12)
    
    plt.tight_layout()
    # savew figure as pdf
    plt.savefig(out_file_name, format='pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    exp_dir_liver = "experiments/reproduction/outputs/liverfailure"
    exp_dir_tramadol = "experiments/reproduction/outputs/tramadol"

    plot_side_by_side(exp_dir_liver, exp_dir_tramadol, out_file_name='plots/train_val_loss_acc.pdf')


    # exp_dir = "experiments/reproduction/outputs/liverfailure"

    # for i in range(1, 6):
    #     visualize_train_val_dynamics(train_result_path=f'{exp_dir}_{i}/logs/train_log.json',
    #                                 val_result_path=f'{exp_dir}_{i}/logs/test_log.json',
    #                                 out_dir=f'plots/liverfailure_{i}', 
    #                                 out_file_name="train_validation_loss_acc",
    #                                 steps_per_epoch=184,
    #                                 plot_title = f"Training and Validation loss & accuracy \n (Analgesic-induced Liver Failure)")

    # visualize_train_val_dynamics(train_result_path='experiments/reproduction/outputs/liverfailure_1/logs/train_log.json',
    #                             val_result_path='experiments/reproduction/outputs/liverfailure_1/logs/test_log.json',
    #                             out_dir="plots/liverfailure_1", 
    #                             out_file_name="train_validation_loss_acc.",
    #                             steps_per_epoch=184,
    #                             plot_title = "Training and Validation loss & accuracy \n (Analgesic-induced Liver Failure)")
    # visualize_train_val_dynamics(train_result_path='experiments/reproduction/outputs/tramadol_1/logs/train_log.json',
    #                             val_result_path='experiments/reproduction/outputs/tramadol_1/logs/val_log.json',
    #                             out_dir="plots/tramadol_1", 
    #                             out_file_name="train_validation_loss_acc",
    #                             steps_per_epoch=128,
    #                             plot_title = "Training and Validation loss & accuracy \n (Tramadol-related mortalities)")