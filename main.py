####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################

import os
import torch
from parser import get_parser
from help import HELP


def main():
    for num_samples in [900, 700, 450, 300, 200, 100, 50]:
        for seed in [1, 2, 3, 4, 42]:
            model = HELP(
                search_space="nasbench201",
                mode="meta-train",
                num_samples=10,
                seed=seed,
                num_meta_train_sample=num_samples,
                # exp_name="reproduce",
                meta_train_devices=[
                    "1080ti_1",
                    "1080ti_32",
                    "1080ti_256",
                    "silver_4114",
                    "silver_4210r",
                    "samsung_a50",
                    "pixel3",
                    "essential_ph_1",
                    "samsung_s7",
                ],
                meta_valid_devices=["titanx_1", "titanx_32", "titanx_256", "gold_6240"],
                meta_test_devices=[
                    "titan_rtx_256",
                    "gold_6226",
                    "fpga",
                    "pixel2",
                    "raspi4",
                    "eyeriss",
                ],
                data_path="/Users/timurcarstensen/PycharmProjects/HELP/data/nasbench201/",
                use_wandb=True,
                project="thesis-help-plots",
                group=f"{num_samples}-samples",
                exp_name=f"seed-{seed}",
            )

            model.meta_train()
            model.test_predictor()
    # if args.mode == "meta-train":
    # elif args.mode == "meta-test":
    #     model.test_predictor()
    #
    # elif args.mode == "nas":
    #     model.nas()


def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def set_gpu(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args


def set_path(args):
    args.data_path = os.path.join(args.main_path, "data", args.search_space)
    args.save_path = os.path.join(args.save_path, args.search_space)
    args.save_path = os.path.join(args.save_path, args.exp_name)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        if args.mode != "nas":
            os.makedirs(os.path.join(args.save_path, "checkpoint"))
    print(f"==> save path is [{args.save_path}] ...")
    return args


if __name__ == "__main__":
    main()
