from utils import common, configuration, dataloader
import vanilla_train

if __name__ == "__main__":

    cfg_json_list = ["config/subj_full.json", "config/subj_50.json", "config/sst2_full.json", "config/sst2_50.json"]

    for cfg_json in cfg_json_list:

        cfg = configuration.config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        train_dataloader = dataloader.get_train_dataloader(cfg)
        test_dataloader = dataloader.get_test_dataloader(cfg)

        vanilla_train.vanilla_finetune_bert(
            cfg,
            train_dataloader,
            test_dataloader,
        )