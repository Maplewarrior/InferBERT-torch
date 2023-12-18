# %%
import pandas as pd
import pdb
import os
import pandas as pd
from utils.misc import load_config
from transformers import AlbertTokenizer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# %% 
def get_data(config_path):
    CFG = load_config(config_path)
    CFG_data = CFG["data"]

    train_path = CFG_data["train_path"]
    val_path = CFG_data["val_path"]
    test_path = CFG_data["test_path"]
    all_path = CFG_data["all_path"]

    df_all = pd.read_csv(all_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    tokenized_data = df_all['sentence'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    return df_all, tokenized_data, df_train, df_val, df_test

# %%
config_path_liverfailure = "configs\liverfailure_config_ALF.yaml"
config_path_tramadol = "configs\\tramadol_config_ALF.yaml"

df_tramadol, tokenized_data_tramadol, df_train_tramadol, df_val_tramadol, df_test_tramdol  = get_data(config_path_tramadol)
df_liverfailure, tokenized_data_liverfailure, df_train_liverfailure, df_val_liverfailure, df_test_liverfailure = get_data(config_path_liverfailure)

# %%
# get length count of all tokens
token_lens_tramadol = [len(token) for token in tokenized_data_tramadol.values]
token_lens_liverfailure = [len(token) for token in tokenized_data_liverfailure.values]

print("Max token length (Tramadol): ", max(token_lens_tramadol))
print("Max token length (Liverfailure): ", max(token_lens_liverfailure))

# %%
# PLOTS
# Function to darken a color
def darken_color(color, factor=0.5):
    # Convert to RGB
    rgb = mcolors.to_rgb(color)
    # Darken each RGB component
    darkened = [c * factor for c in rgb]
    return darkened

# Base color for the face of the bins
base_color_tramadol = 'bisque'
base_color_liverfailure = 'lightblue'

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(6, 4))

# Histogram for Tramadol-related mortalities
axs[0].hist(token_lens_tramadol, bins=30, edgecolor=darken_color(base_color_tramadol, factor=0.5), facecolor=base_color_tramadol, label="Tramadol-related mortalities")
# axs[0].set_ylabel('Number of sentences')
axs[0].legend()

# Histogram for the other dataset
axs[1].hist(token_lens_liverfailure, bins=30, edgecolor=darken_color(base_color_liverfailure, factor=0.5), facecolor=base_color_liverfailure, label="Analgesics-induced acute liver failure")
axs[1].set_xlabel('Token count per sentence')
# axs[1].set_ylabel('Number of sentences')
axs[1].legend()

fig.suptitle('Distribution of sentence length')

# Make a ylabel for the entire figure
fig.text(0.0, 0.5, 'Number of sentences', va='center', rotation='vertical')


# Adjust layout
plt.tight_layout()

# plot savefig to file pdf with width in cm
fig.savefig('plots/token_length_distribution.pdf', bbox_inches='tight', pad_inches=0.1, format='pdf', dpi=300)


# %%

def get_line(df):
    count = df['label'].value_counts()
    perc_pos = count[1] / (count[0] + count[1])
    num_samples = len(df)
    ratio = count[1]/count[0]
    print(f"  N: {num_samples}, Positive: {perc_pos*100:.2f}%, Ratio: {ratio:.2f}")
    return num_samples, perc_pos, ratio
    print()

print("Liverfailure:")
df = df_liverfailure
df_train = df_train_liverfailure
df_val = df_val_liverfailure
df_test = df_test_liverfailure

# count number number of samples

print("Total:")
get_line(df)
print("Train:")
get_line(df_train)
print("Val:")
get_line(df_val)
print("Test:")
get_line(df_test)

print()
print("Tramadol:")
df = df_tramadol
df_train = df_train_tramadol
df_val = df_val_tramadol
df_test = df_test_tramdol

print("Total:")
get_line(df)
print("Train:")
get_line(df_train)
print("Val:")
get_line(df_val)
print("Test:")
get_line(df_test)
# %%
