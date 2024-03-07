import torch

def mean_l1_norm_by_bins(label, prediction, bin_edges: list = [-1, -0.6, -0.2, 0.2, 0.6, 1]):
    # Ensure label and prediction are properly shaped and on the same device
    assert label.shape == prediction.shape, "label and prediction must have the same shape"
    
    # Define the bin edges
    bin_edges = torch.tensor(bin_edges, device=label.device)
    
    # Calculate the mean L1 norm for each bin
    means = []
    for i in range(len(bin_edges)-1):
        # Find indices of labels that fall into the current bin
        indices = (label >= bin_edges[i]) & (label < bin_edges[i+1])
        
        # Calculate the L1 norm for these indices
        l1_norm = torch.abs(label[indices] - prediction[indices])
        
        # Calculate the mean of the L1 norm for the current bin
        mean_l1 = torch.mean(l1_norm)
        means.append(mean_l1)
    
    # Convert the list of means to a tensor
    means = torch.stack(means)
    
    # Optionally, calculate the overall mean across bins
    overall_mean = torch.mean(means)
    
    return means, overall_mean, bin_edges

# Example usage
batch_size = 10
sequence_length = 20
label = torch.randn(batch_size, 1, sequence_length) * 2 / 3 - 1/3  # Example label tensor
prediction = torch.randn(batch_size, 1, sequence_length) * 2 / 3 - 1/3  # Example prediction tensor

means, overall_mean = mean_l1_norm_by_bins(label, prediction)
print("Mean L1 Norm by Bins:", means)
print("Overall Mean L1 Norm:", overall_mean)
