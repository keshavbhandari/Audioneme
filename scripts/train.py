import torch.nn.functional as F
import torch.optim as optim
# from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

from src.models.audiomer import AudiomerClassification as Audiomer
from scripts.data_loader import train_loader, val_loader, test_loader
from src.utils.data import get_digits
from src.models.resnetse34v2.resnetse34v2 import ResNetSE34V2
from src.models.resnetse34v2.resnetse34v2_classifier import ResNetSE34V2_Classification
from src.training.training_functions import count_parameters, EarlyStopping
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

        data, file = data
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

        data, file = data
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

        data, file = data
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

if model_type == "audiomer":
    config = [1, 4, 8, 8, 16, 16, 32, 32, 64, 64]  # Audiomer S - 180K
    input_size = 8192
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

else:
    pretrained_model = ResNetSE34V2(n_bins=n_bins)
    # pretrained_model.load_state_dict(torch.load(PRETRAINED_ResnetSE34V2))
    model = ResNetSE34V2_Classification(pretrained_model).to(device)

# for n, param in enumerate(model.parameters()):
#     if n < 150:
#         param.requires_grad = False

count_parameters(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=320)

print('INFO: Initializing early stopping')
early_stopping = EarlyStopping(patience=early_stopping_rounds)

pbar_update = round(1 / (len(train_loader) + len(val_loader)), 4)
train_loss, val_loss, test_loss = [], [], []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(train_loader, model, device, epoch, log_interval, accum_iter=2)
        validate(val_loader, model, device, epoch)
        # Save Best Model
        if epoch == 1:
            print('Saving model...')
            torch.save(model, "speech_command_recognition.pth")
        else:
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


tokenizer_path = DATA_DIR / 'test/tokenizer.pt'
tokenizer = torch.load(tokenizer_path)
encoder_tokenizer = tokenizer['encoder']
decoder_tokenizer = tokenizer['decoder']

# device = 'cpu'
actuals = []
predicted = []
filename = []
for i, (data, target) in enumerate(test_loader):
    data, file = data
    data = data.to(device)
    target = target.to(device)
    best_model = best_model.to(device)
    output = best_model(data)
    output = torch.sigmoid(output)
    pred = torch.squeeze(output, -1)
    predicted += pred.detach().cpu().numpy().tolist()
    actuals += target.detach().cpu().numpy().tolist()
    filename.append(file)
filename = torch.cat(filename)
filename = [decoder_tokenizer[str(file.detach().numpy().tolist())] for file in filename]

results = pd.DataFrame({
    'filename': filename,
    'actuals': actuals,
    'predicted': predicted
})
results['participant'] = results['filename'].apply(lambda x: x.split('/')[-1].split('-')[0:2])#.str.extract('([0-9]+)', expand=False)#.astype(int)
results['participant'] = results['participant'].apply(lambda x: x[0] + '_' + str(get_digits(x[1])))
results['predicted_disorder'] = np.where(results['predicted']>0.5, 1, 0)
aggregated = results.groupby(by=['participant', 'actuals'], as_index=False).agg({'predicted_disorder': ['sum', 'count'], 'predicted': ['mean', 'median', 'min', 'max', 'std']}).droplevel(axis=1, level=0).reset_index(drop=True)
aggregated['pct'] = round(aggregated['sum'] / aggregated['count'], 4)
print(aggregated)

y_pred = [round(elem) for elem in predicted]
print(classification_report(actuals, y_pred))

cm1 = confusion_matrix(actuals, y_pred)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('specificity : ', specificity1 )

sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('sensitivity : ', sensitivity1)

plt.plot(train_loss);
plt.plot(val_loss);
plt.title("training and val loss");