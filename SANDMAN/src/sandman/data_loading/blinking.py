import torch
from torch.utils.data import Dataset, DataLoader

class BlinkingToyDataset(Dataset):
    def __init__(self, T: int = 100, device="cpu"):
        """
        T: number of timesteps
        """
        self.T = T
        self.device = device

        # neurons: [A, B, C, D, E, F, G, H]
        self.neuron_regions = torch.tensor(
            [0, 0, 1, 1, 2, 2, 2, 2],  # [N]
            dtype=torch.long,
            device=device
        )

        self.eid = torch.tensor(0, dtype=torch.long, device=device)

    def __len__(self):
        return 1000  # arbitrary

    def __getitem__(self, idx):
        T = self.T
        device = self.device

        t = torch.arange(T, device=device)

        # -------- Region 0 (fast blinking) --------
        phase = torch.randint(0, 2, (), device=device)
        A = (t + phase) % 2
        Bn = 1 - A

        # -------- Region 1 (slow blinking) --------
        C = ((t // 2) % 2)
        D = 1 - C

        # -------- Region 2 (copies) --------
        E = A.clone()
        F = Bn.clone()
        G = C.clone()
        H = D.clone()

        spikes = torch.stack(
            [A, Bn, C, D, E, F, G, H],
            dim=-1
        ).float()          # [T, N]

        return {
            "spikes_data": spikes,                 # [T, N]
            "neuron_regions_full": self.neuron_regions,  # [N]
            "eid": self.eid,                       # scalar
        }


def make_blinking_toy_loader(T: int, batch_size: int, device="cpu"):

    dataset = BlinkingToyDataset(T=T, device=device)

    def collate_fn(batch):
        return {
            "spikes_data": torch.stack([b["spikes_data"] for b in batch], dim=0),        # [B, T, N]
            "neuron_regions_full": torch.stack([b["neuron_regions_full"] for b in batch], dim=0),  # [B, N]
            "eid": torch.stack([b["eid"] for b in batch], dim=0),                        # [B]
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return {
        "train": loader,
        "test": loader,
        "val": loader,
    }


if __name__ == "__main__":
    loader = make_blinking_toy_loader(T=16, batch_size=2)
    batch = next(iter(loader["train"]))

    for k, v in batch.items():
        print(k, v.shape, v.dtype)

