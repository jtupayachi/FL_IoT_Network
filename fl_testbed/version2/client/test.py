""".Xauthority

python3 fl_testbed/version2/client/test.py

"""


import pandas as pd
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import os

def split_dataframe_to_non_identical_clients(df, N, alpha, num_clients, num_samples_per_client, p, output_dir):
    if alpha <= 0:
        raise ValueError("Alpha must be a positive number.")
    if len(p) != N or not all(0 <= qi <= 1 for qi in p):
        raise ValueError("The prior class distribution 'p' must be a 1-dimensional array with values between 0 and 1.")
    
    clients = []
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

    for i in range(num_clients):
        num_samples = min(num_samples_per_client, len(df))
        data = df.iloc[:num_samples, :].copy()
        df = df.iloc[num_samples:, :]

        q = dirichlet.rvs([alpha] * N)  # Ensure alpha is a scalar
        q = np.maximum(q, 0)
        q /= q.sum()

        data['label'] = [np.random.choice(N, p=np.squeeze(q)) for _ in range(num_samples)]
        client_info = {
            'q': q,
            'data': data
        }
        clients.append(client_info)

    return clients

def plot_class_distribution_and_save(client, output_dir, client_index):
    q = client['q']
    plt.bar(range(len(q)), np.squeeze(q))
    plt.xticks(range(len(q)), range(1, len(q) + 1))
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Class Distribution (q) for Client {}'.format(client_index))
    
    # Save the plot as an image file
    output_file = os.path.join(output_dir, 'client_{}_class_distribution.png'.format(client_index))
    plt.savefig(output_file)
    plt.close()  # Close the plot to avoid displaying it

# Example usage:
N = 5  # Number of classes
alpha = 0.001  # Concentration parameter (a scalar)
num_clients = 5
num_samples_per_client = 100
p = [0.2, 0.2, 0.2, 0.2, 0.2]  # Prior class distribution (1-dimensional)

# Create a sample DataFrame with 'x' and 'y' columns
data = {'x': np.random.randn(500), 'y': np.random.randn(500)}
df = pd.DataFrame(data)

# Output directory for saving plots
output_directory = "client_plots"
os.makedirs(output_directory, exist_ok=True)

non_identical_clients = split_dataframe_to_non_identical_clients(df, N, alpha, num_clients, num_samples_per_client, p, output_directory)

# Generate and save plots for each client
for i, client in enumerate(non_identical_clients):
    print(f"Client {i + 1} Class Distribution (q):")
    print(client['q'])
    plot_class_distribution_and_save(client, output_directory, i)

print("Plots have been saved to the '{}' directory.".format(output_directory))
