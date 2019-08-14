import torch
import numpy as np
import os, sys
os.environ['TORCH_MODEL_ZOO'] = '/mnt/projects/counting/pretrained/resnet'
import models, datasets, metrics
import utils as ut
import tqdm, time
import pandas as pd


def main():
    exp_dict = {"model":"MRCNN",
                "max_epochs":10,
                "batch_size":1}

    model = models.mrcnn.MRCNN(exp_dict).cuda()

    path_base = "checkpoints"
    # model_state_dict = torch.load(path_base + "/model_state_dict.pth")
    # model.load_state_dict(model_state_dict)

    train_set = datasets.pascal2012.Pascal2012(split="train", exp_dict=exp_dict,
                                               root="/mnt/datasets/public/issam/VOCdevkit/VOC2012/")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=train_set.collate_fn,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True)

    # Main loop
    model.history = {"score_list": []}

    for e in range(exp_dict["max_epochs"]):
        # Train for one epoch
        score_dict = train_epoch(model, train_loader)
        score_dict["epoch"] = e

        # Update history
        model.history["score_list"] += [score_dict]

        # Report
        results_df = pd.DataFrame(model.history["score_list"]).tail()
        print(results_df[["epoch", "train_loss"]])

    # Experiment completed
    torch.save("./checkpoints/checkpoint.pth", {"model_state_dict": model.state_dict(),
                                                "history": model.history})

def train_epoch(model, train_loader):
    """Trainer."""
    model.train()

    # Init variables
    n_batches = len(train_loader)
    loss_sum = 0.
    s_time = time.time()

    # Start training
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(train_loader):
        # Compute loss
        loss = model.train_step(batch)
        if np.isnan(loss):
            raise ValueError('loss=NaN ...')
        loss_sum += loss
        pbar.set_description("Training loss: %.4f" % (loss_sum / (i + 1)))
        pbar.update(1)
    pbar.close()

    # Update history
    e_time = time.time()
    score_dict = {
        "train_loss": loss_sum / n_batches,
        "train_time_taken": e_time - s_time,
    }

    return score_dict

if __name__ == "__main__":
    main()
