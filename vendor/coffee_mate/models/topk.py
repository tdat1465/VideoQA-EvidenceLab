import torch

def HardtopK(x, k):
    B, L = x.size()
    indices = torch.topk(x, k=k, dim=-1, sorted=False).indices
    indices, _ = torch.sort(indices, descending=False, dim=-1)
    idx = x.new_zeros(B, L, k)
    idx[torch.arange(B).unsqueeze(1), indices, torch.arange(k).unsqueeze(0)] = 1
    return idx.half()

def HardtopK_segment(x, m=8, k=1):
    B, L = x.size()
    x_reshaped = x.view(B, m, L//m)
    indices = torch.topk(x_reshaped, k=k, dim=-1, sorted=False).indices
    indices, _ = torch.sort(indices, descending=False, dim=-1)
    if m == 8:
        indices = indices.view(B, -1) + L//m * torch.Tensor([int(0/k), int(1/k), int(2/k), int(3/k),
                                                             int(4/k), int(5/k), int(6/k), int(7/k)]).to(x.device)
    elif m == 4:
        indices = indices.view(B, -1) + L//m * torch.Tensor([int(0/k), int(1/k), int(2/k), int(3/k)]).to(x.device)
    indices = indices.long()
    idx = x.new_zeros(B, L, m*k)
    idx[torch.arange(B).unsqueeze(1), indices, torch.arange(m*k).unsqueeze(0)] = 1
    return idx.half()

