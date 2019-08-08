import torch
import os, sys
os.environ['TORCH_MODEL_ZOO'] = '/mnt/projects/counting/pretrained/resnet'
import models, datasets, metrics
import utils as ut
import tqdm, time

def main():
    exp_dict = {"model":"MRCNN",
                "option":{"batch_size":1,
                          "pointList":"prm",
                             "backbone":"FPN_R50",
                             "proposal_name":"sm",
                             "apply_void":"True",
                             "lr":"mrcnnn"}}
    model = models.mrcnn.MRCNN(exp_dict).cuda()

    path_base = "checkpoints"
    model_state_dict = torch.load(path_base + "/model_state_dict.pth")
    model.load_state_dict(model_state_dict)

    val_set = datasets.pascal2012.Pascal2012Original(split="val", exp_dict=exp_dict)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        collate_fn=val_set.collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    model.eval()

    # Init variables
    n_batches = len(val_loader)
    metric_class = metrics.ap.AP50_segm
    metric_object = metric_class()
    metric_name = type(metric_object).__name__

    # Validate
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(val_loader):
        # Validate a batch
        metric_object.add_batch(model, batch)
        score_dict = metric_object.get_score_dict()

        pbar.set_description("  > Validation %s: %.4f" % (metric_name, score_dict["score"]))
        pbar.update(1)
    pbar.close()

    # score should be around 40.4
    print("score:", metric_object.get_score_dict())

if __name__ == "__main__":
    main()
