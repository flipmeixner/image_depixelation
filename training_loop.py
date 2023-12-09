import numpy as np
import torch
from torch.utils import data
from typing import Tuple
import matplotlib.pyplot as plt
from stack_with_padding import stack_with_padding
from torch.backends import mps
from tqdm import tqdm


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        show_progress: bool = False
) -> Tuple[list, list]:
    optimizer = torch.optim.Adam(params=network.parameters(), lr=1e-5, weight_decay=1e-5)
    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        collate_fn=stack_with_padding,
        num_workers=10,
        pin_memory=True
    )
    data_loader_eval = torch.utils.data.DataLoader(
        dataset=eval_data,
        batch_size=64,
        collate_fn=stack_with_padding,
        num_workers=10,
        pin_memory=True
    )
    # cuda can be added if executed on machine with cuda available
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    network.to(device)
    epoch_losses = []
    eval_losses = []
    # init progress bar
    progress_bar = None
    if show_progress:
        total_iterations = num_epochs * (len(data_loader_train) + len(data_loader_eval))
        progress_bar = tqdm(total=total_iterations, desc="Training and Evaluation")
    for epoch in range(num_epochs):
        mbl = []
        network.train()
        for inputs, known, targets, _ in data_loader_train:
            inputs = inputs.to(device).float()
            known = known.to(device).float()
            targets = targets.to(device).float()
            optimizer.zero_grad()
            output = network(inputs)
            known_tensor = known.bool()
            # Extract only pixelated area from output / targets for loss calculation
            target = targets[known_tensor]
            out = output[known_tensor]
            rmse_loss = torch.nn.MSELoss()(out, target)
            rmse_loss = torch.sqrt(rmse_loss)
            rmse_loss.backward()
            optimizer.step()

            mbl.append(rmse_loss.item())
            if show_progress:
                progress_bar.update(1)
        epoch_losses.append(np.mean(mbl))
        # Save model state after each epoch, can be optimized, good for now
        torch.save(network.state_dict(), f"model_{epoch + 1}.pth")

        network.eval()
        with torch.no_grad():
            eval_mlb = []
            for inputs, known, targets, _ in data_loader_eval:
                inputs = inputs.to(device).float()
                known = known.to(device).float()
                targets = targets.to(device).float()
                output = network(inputs)
                known_tensor = known.bool()
                target = targets[known_tensor]
                out = output[known_tensor]
                loss = torch.nn.MSELoss()(out, target)
                loss = torch.sqrt(loss)
                eval_mlb.append(loss.item())
                if show_progress:
                    progress_bar.update(1)
            eval_losses.append(np.mean(eval_mlb))
    return epoch_losses, eval_losses


if __name__ == "__main__":
    from model import SimpleCNN
    from create_dataset import RandomImagePixelationDataset

    torch.random.manual_seed(0)
    # training_raw dir contains 300 folders from provided images and one folder with all images from these 300 folders,
    # but augmented (sums up to about 60k images). (also see file image_augmentation)
    train_data = RandomImagePixelationDataset(
        r"training_raw",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16)
    )
    # eval_raw dir contains the remaining 50 folders for evaluation (~5k images)
    eval_data = RandomImagePixelationDataset(
        r"eval_raw",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16)
    )
    # working config: in_channels=1, out_channels=128, hidden_channels=6, kernel_size=7, dropout_rate=0.01
    network = SimpleCNN(in_channels=1, out_channels=128, hidden_channels=6, kernel_size=7, dropout_rate=0.01)

    model_param = filter(lambda p: p.requires_grad, network.parameters())
    params = sum(np.prod(p.size()) for p in model_param)
    print(f"Model has {params} parameters.")

    epochs, train_loss, eval_loss = [], [], []
    train_losses, eval_losses = training_loop(network, train_data, eval_data,
                                              num_epochs=20, show_progress=True)
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")
        epochs.append(epoch)
        train_loss.append(tl)
        eval_loss.append(el)

    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, eval_loss, label='eval')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.show()
