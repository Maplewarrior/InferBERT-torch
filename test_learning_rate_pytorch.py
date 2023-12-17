
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy model for demonstration
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 2)  # Simple linear layer

    def forward(self, x):
        return self.linear(x)

def build_lr_scheduler(optimizer, num_train_steps, num_warmup_steps):
    return get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps=num_warmup_steps,
                                           num_training_steps=num_train_steps)

def build_optimizer(model, lr):
    weight_decay_rate = 0.0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-6

    betas = (beta_1, beta_2)

    optimizer = optim.AdamW(params=model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay_rate,
                            betas=betas,
                            eps=epsilon)
    return optimizer

# Set up the optimizer
num_train_steps = 10000
num_warmup_steps = 200
init_lr = 2.0e-5

# Create dummy model and optimizer
model = DummyModel()
optimizer = build_optimizer(model, init_lr)

scheduler = build_lr_scheduler(optimizer, num_train_steps, num_warmup_steps)
# Calculate learning rates for PyTorch
lrs_pytorch = []
for _ in range(num_train_steps):
    optimizer.step()
    lrs_pytorch.append(scheduler.get_last_lr()[0])
    scheduler.step()

# load learning rates from file
lrs_tf = []
with open(f'lrs_init_lr_{init_lr}_num_train_steps_{num_train_steps}_num_warmup_steps_{num_warmup_steps}.txt', 'r') as f:
    for line in f:
        lrs_tf.append(float(line.strip()))

# Plot the learning rate
plt.plot(lrs_tf, label="TensorFlow")
plt.plot(lrs_pytorch, label="PyTorch")
plt.xlabel("Step")
plt.ylabel("Learning rate")
plt.title(f"Learning rate schedule. \n Init LR: {init_lr}, Num train steps: {num_train_steps}, Num warmup steps: {num_warmup_steps}")
# grid lines
plt.legend()
plt.grid(True)

# save plot to file
plt.show()
        