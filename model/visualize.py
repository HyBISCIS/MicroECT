# Copyright (c) 2023, HyBISCIS Team (Brown University, Boston University)
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

import os 
import numpy as np
import matplotlib.pyplot as plt

def draw_curve(epoch, train_loss, val_loss, loss, save_path):
    epochs = np.arange(0, len(train_loss), 1)
    plt.plot(epochs, train_loss, 'bo-', label='train')
    plt.plot(epochs, val_loss, 'ro-', label='val')
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss {loss}")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, 'train.png'))
    plt.close()


def draw_pred_grid(ground_truth, predictions, save_path):
    methods = list(predictions.keys())
    num_rows = ground_truth.shape[0]

    num_cols = len(methods)

    fig, axs = plt.subplots(num_rows, num_cols+1)

    # plot the ground truth
    for i in range(num_rows):
        axs[i, 0].imshow(ground_truth[i][0])

    # plot the predictions 
    for j in range(num_cols): 
        preds = predictions[methods[j]]
        for i in range(num_rows):
            axs[i, j+1].imshow(preds[i][0])
            axs[i, j+1].set_title(f"{methods[j]} Predictions")

    plt.savefig(save_path)
    plt.close()