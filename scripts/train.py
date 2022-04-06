import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from src.models.audiomer import AudiomerClassification as Audiomer
from src.training.early_stopping import EarlyStopping
from scripts.data_loader import train_loader, val_loader, test_loader
from configs.constants import *


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def train(train_loader, model, device, epoch, log_interval, accum_iter=1):
    model.train()
    correct = 0
    train_running_loss = 0.0
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        counter += 1

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)
        output = torch.sigmoid(output)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.binary_cross_entropy(output, target.reshape(-1, 1).to(
            torch.float32))  # output.squeeze() --> (batch x n_output)
        loss = loss / accum_iter

        # pred = get_likely_index(output)
        correct += number_of_correct(output.squeeze(-1).round(), target)

        loss.backward()
        # weights update
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader.dataset)):
            optimizer.step()
            optimizer.zero_grad()

        # print training stats
        if batch_idx % log_interval == 0:
            print \
                (f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        train_running_loss += loss.item()

    train_loss.append(train_running_loss / counter)
    print \
        (f"\nTrain Epoch: {epoch} \tLoss: {train_running_loss / counter:.6f} \tAccuracy: {correct}/{len(train_loader.dataset)} ({100. * correct / len(train_loader.dataset):.0f}%)\n")


def validate(val_loader, model, device, epoch):
    model.eval()
    correct = 0
    val_running_loss = 0.0
    counter = 0
    for data, target in val_loader:
        counter += 1

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)
        output = torch.sigmoid(output)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.binary_cross_entropy(output, target.reshape(-1, 1).to(torch.float32))  # output --> (batch x n_output)

        # pred = get_likely_index(output)
        correct += number_of_correct(output.squeeze(-1).round(), target)

        # update progress bar
        pbar.update(pbar_update)
        val_running_loss += loss.item()

    val_loss.append(val_running_loss / counter)
    print \
        (f"\nVal Epoch: {epoch} \tLoss: {val_running_loss / counter:.6f} \tAccuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n")


def test(test_loader, model, device, epoch):
    model.eval()
    correct = 0
    test_running_loss = 0.0
    counter = 0
    for data, target in test_loader:
        counter += 1

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        output = model(data)
        output = torch.sigmoid(output)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.binary_cross_entropy(output, target.reshape(-1, 1).to(
            torch.float32))  # output.squeeze() --> (batch x n_output)

        # pred = get_likely_index(output)
        correct += number_of_correct(output.squeeze(-1).round(), target)

        # update progress bar
        pbar.update(pbar_update)
        test_running_loss += loss.item()

    test_loss.append(test_running_loss / counter)
    print \
        (f"\nTest Epoch: {epoch} \tLoss: {test_running_loss / counter:.6f} \tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

config = [1, 4, 8, 8, 16, 16, 32, 32, 64, 64]  # Audiomer S - 180K
input_size = 8192 * 1
model = Audiomer(
        input_size=input_size,
        config=config,
        kernel_sizes=[5] * (len(config) - 1),
        num_classes=1,
        depth=1,
        num_heads=2,
        pool="cls",
        mlp_dim=config[-1],
        mlp_dropout=0.2,
        use_residual=True,
        dim_head=32,
        expansion_factor=2,
        use_attention=True,
        use_se=True,
        equal_strides=False
    ).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=320)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

print('INFO: Initializing early stopping')
early_stopping = EarlyStopping(patience=3)

pbar_update = round(1 / (len(train_loader) + len(val_loader)), 4)
train_loss, val_loss, test_loss = [], [], []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(train_loader, model, device, epoch, log_interval, accum_iter=2)
        validate(val_loader, model, device, epoch)
        # Save Best Model
        if len(val_loss) > 1:
            if val_loss[-1] < min(val_loss[:-1]):
                print('Saving model...')
                torch.save(model, "speech_command_recognition.pth")
        # Early Stopping
        early_stopping(val_loss[-1])
        if early_stopping.early_stop:
            break
        scheduler.step()
    best_model = torch.load("speech_command_recognition.pth")
    test(test_loader, best_model, device, epoch)
