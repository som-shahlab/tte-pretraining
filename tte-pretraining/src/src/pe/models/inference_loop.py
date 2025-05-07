
import torch
from torch.utils.tensorboard import SummaryWriter

class VisionEncoder:
    def __init__(self,):
        self.writer        =  SummaryWriter()



def train_loop(self, model, optimizer, loss_function, train_ds, val_loader, train_loader, 
               device, max_epochs: int = 10, val_interval: int = 2, save_path: str = None):
    ## Start Logg Values ##
    best_metric       = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values     = []

 

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        self.model.train()
        epoch_loss,step = 0,0
    
        for batch_data in self.train_loader:
            ### Forward Loop ###
            step          += 1
            inputs, labels = batch_data[0].to(device) , batch_data[1].to(device)
            self.optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_function(outputs, labels)

            ### Backward Loop ###
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len   = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            self.writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    self.writer.close()