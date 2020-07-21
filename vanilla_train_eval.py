from utils import common, configuration, dataloader
import bert

if __name__ == "__main__":

    cfg_json_list = [
        "config/vanilla/trec_50.json",
        "config/vanilla/trec_full.json",
        "config/vanilla/imdb_50.json",
        "config/vanilla/imdb_full.json",
        "config/vanilla/sst2_50.json",
        "config/vanilla/sst2_full.json",
        "config/vanilla/subj_50.json",
        "config/vanilla/subj_full.json",
        ]

    for cfg_json in cfg_json_list:

        cfg = configuration.config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        train_dataloader = dataloader.get_train_dataloader(cfg)
        test_dataloader = dataloader.get_test_dataloader(cfg)

        bert.vanilla_finetune_bert(
            cfg,
            train_dataloader,
            test_dataloader,
        )