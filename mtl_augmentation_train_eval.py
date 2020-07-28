from utils import common, configuration, dataloader
import bert_mtl

if __name__ == "__main__":

    cfg_json_list = [
        "config/mtl_aug/subj_20.json",
        # "config/mtl_aug/sst2_20.json",
        ]

    for cfg_json in cfg_json_list:

        cfg = configuration.config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        train_dataloader = dataloader.get_augmented_train_dataloader(cfg)
        test_dataloader = dataloader.get_test_dataloader(cfg)

        bert_mtl.mtl_aug_finetune_bert(
            cfg,
            train_dataloader,
            test_dataloader,
        )