import torch
import numpy as np


def inference(config, model, testloader):
    class_correct = list(0.0 for _ in range(10))
    class_total = list(0.0 for _ in range(10))
    total_correct = 0
    total_samples = 0
    val_scores = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(config["device"]), labels.to(config["device"])
            outputs = model(images)
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
        print("Accuracy of %5s : %2d %%" % (config["classes"][i], accuracy))

    val_score = np.mean(val_scores)
    print("Overall Accuracy: %.2f %%" % val_score)
    return val_score
