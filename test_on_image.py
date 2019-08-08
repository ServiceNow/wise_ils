import torch
import os, sys
os.environ['TORCH_MODEL_ZOO'] = '/mnt/projects/counting/pretrained/resnet'
import models, datasets, metrics
import utils as ut
import tqdm, time

def main():
    print("running mask rcnn on a validation image...")

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
    batch = val_set.collate_fn([val_set[0]])
    vis_dict = model.visualize(batch)
    for k in vis_dict:
        ut.save_image("results/%d_%s.png" % (batch["meta"]["index"][0].item(), k), vis_dict[k])

    print("images saved in ./results")

if __name__ == "__main__":
    main()
