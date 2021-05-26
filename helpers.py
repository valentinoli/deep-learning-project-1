def create_dataloader(*tensors, batch_size = 10, shuffle = True):
    """Creates a PyTorch data loader from the given tensors"""
    dataset = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
