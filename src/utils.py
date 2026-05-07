import torch
from torch import nn

# =========================
# Weight initialization
# =========================
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # Kaiming Normal (He Normal)
        nn.init.kaiming_normal_(m.weight,mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Kaiming Normal (He Normal)
        nn.init.kaiming_normal_(m.weight,mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# =========================
# Train loop
# =========================
def train_loop(model,optimizer,loss_fn,dataloader,device,train_acc_list=None, train_loss_list=None):
  model.train()
  if train_acc_list is None:
    train_acc_list = []
  if train_loss_list is None:
    train_loss_list = []  

  current=0
  size = len(dataloader.dataset)
  for batch, (x_train,y_train) in enumerate(iter(dataloader)):
    x_train,y_train = x_train.to(device),y_train.to(device)
    logits = model(x_train)

    optimizer.zero_grad()
    loss = loss_fn(logits,y_train)
    loss.backward()
    optimizer.step()

    current += (logits.argmax(dim=1) == y_train).sum()

  train_acc = current/size
  print(f"train accuracy: {train_acc} / train loss: {loss.item()} ")
  train_acc_list.append(train_acc.item())
  train_loss_list.append(loss.item())

# =========================
# Test loop
# =========================
def test_loop(model,dataloader,device,test_acc_list=None):
  model.eval()
  if test_acc_list is None:
    test_acc_list=[]

  current=0
  size = len(dataloader.dataset)
  for batch, (x_test,y_test) in enumerate(iter(dataloader)):
    x_test,y_test = x_test.to(device),y_test.to(device)

    with torch.no_grad():
      logits = model(x_test)
      current += (logits.argmax(dim=1) == y_test).sum()
  test_acc = current/size
  print(f"test accuracy: {test_acc}")
  test_acc_list.append(test_acc.item())
