#SIMILAR IMPLEMENTATION AS: arXiv:1909.06335


import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import os

root_path = os.path.dirname(os.path.abspath("__file__"))



DATA_FOLDER = "/FL_AM_Defect-Detection/fl_testbed/version2/client/"
os.chdir(root_path+DATA_FOLDER)
print(os.getcwd)

def generate_non_identical_clients(N, alpha, num_clients, num_samples_per_client, p):
    clients = []

    for _ in range(num_clients):
        q = dirichlet.rvs(alpha * np.array(p))
        q = np.maximum(q, 0)
        q /= q.sum()

        data = []
        for _ in range(num_samples_per_client):
            # Generate class labels following a categorical distribution (q)
            sample = np.random.choice(N, p=np.squeeze(q))
            data.append(sample)

        client_info = {
            'q': q,
            'data': data
        }
        clients.append(client_info)

    return clients

def plot_class_distribution(client,name):
    q = client['q']
    plt.bar(range(len(q)), np.squeeze(q))
    plt.xticks(range(len(q)), range(1, len(q) + 1))
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Class Distribution (q) for a Client')
    plt.savefig("Client: " + str(name))

# Example usage and visualization:
N = 5  # Number of classes
alpha = 1.0  # Concentration parameter
num_clients = 3
num_samples_per_client = 100
p = [0.2, 0.2, 0.2, 0.2, 0.2]  # Prior class distribution

non_identical_clients = generate_non_identical_clients(N, alpha, num_clients, num_samples_per_client, p)

# Visualize the class distribution for the first client
for i, client in enumerate(non_identical_clients):
    print(f"Client {i + 1} Class Distribution (q):")
    plot_class_distribution(client,i)
