import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from UNetModel import UNet
from DataLoading import CityScapesDataset, CityscapesDataLoader,train_dataloader,val_dataloader , train_dataset , val_dataset
from tqdm import tqdm


LEARNING_RATE=1e-4
BATCH_SIZE=4
NUM_EPOCHS=10
NUM_CLASSES=19
SAVE_PATH= "utils/model.pth"
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

model=UNet(n_channels=3 , n_classes=NUM_CLASSES)
model.to(DEVICE)


criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters() , lr=LEARNING_RATE)

def train_fn(loader ,model ,optimizer,criterion,scaler=None):
    model.train()
    running_loss=0.0
    loop=tqdm(loader , desc="Training" , leave=True)

    for images , labels in loop:
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)

        # Zero Gradinets to start the training step
        optimizer.zero_grad()
        # forward pass
        predictions=model(images)
        #calulating loss
        loss=criterion(predictions , labels)
        #backward pass
        loss.backward()
        #optimizing 
        optimizer.step()

        running_loss+=loss.item()
        loop.set_postfix(loss=running_loss/len(loader))

        return running_loss/len(loader)


def check_validation(loader , model , criterion):
    model.eval()
    running_val_loss=0.0
    with torch.no_grad():
        loop=tqdm(loader , desc="Validation" , leave=True)
        for images , labels in loop :
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            #forward pass
            predictions=model(images)
            loss=criterion(predictions , labels)

            running_val_loss+=loss.item()
            loop.set_postfix(val_loss=running_val_loss/len(loader)) 

            return running_val_loss/len(loader)

best_val_loss=float("inf")

for epochs in range(NUM_EPOCHS):
    print(f"Epoch [{epochs+1}/{NUM_EPOCHS}]")
    train_loss=train_fn(train_dataloader , model , optimizer , criterion)
    val_loss=check_validation(val_dataloader , model , criterion)

    if val_loss<best_val_loss:
        best_val_loss=val_loss
        torch.save(model.state_dict() , SAVE_PATH)
        print(f"Model saved with validation loss: {best_val_loss:.4f}")


    