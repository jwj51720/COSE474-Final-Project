import torch
import numpy as np
from tqdm import tqdm


def validation(config, model, validloader, criterion):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    total_correct = 0
    total_samples = 0
    val_scores = []
    model.eval()
    print("**validation**")
    valid_loss = 0
    for i, data in tqdm(enumerate(validloader)):
        images, labels = data
        images, labels = images.to(config["device"]), labels.to(config["device"])
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total_correct += c.sum().item()
        total_samples += labels.size(0)
        for i in range(labels.size(0)):  # 모든 이미지에 대해 반복
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
        val_scores.append(accuracy)

    val_loss = valid_loss / len(validloader)
    val_score = np.mean(val_scores)
    return val_loss, val_score


def trainer(config, fold_loaders, model, writer, criterion, optimizer, scheduler):
    trainloader, validloader = fold_loaders
    min_val_loss = float("inf")
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        print(f"**Epoch {epoch} Training**")
        model.train()
        total_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(trainloader)
        val_loss, val_score = validation(config, model, validloader, criterion)
        if val_loss < min_val_loss:
            print("New Minimum Valid Loss!")
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"{config['save_path']}/current_best.pt")
        print("- Train CrossEntropy Loss: ", train_loss)
        print("- Valid CrossEntropy Loss: ", val_loss)
        print("- Valid Overall Accuracy: ", val_score)
        writer.add_scalar("Train/Loss", train_loss, global_step=epoch)
        writer.add_scalar("Valid/Loss", val_loss, global_step=epoch)
        writer.add_scalar("Valid/Score", val_score, global_step=epoch)
    print("Finished Training")
    return model
