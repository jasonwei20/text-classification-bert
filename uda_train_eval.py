from utils import common, configuration, dataloader
import bert

if __name__ == "__main__":

    cfg_json_list = [
        "config/uda/imdb_20_uda_backtranslation_sharpen08.json",
        # "config/uda/imdb_20_uda_sr.json",
        # "config/uda/imdb_20_uda_swaps.json",
        # "config/uda/imdb_20_no_uda.json",
        ]

    for cfg_json in cfg_json_list:

        cfg = configuration.uda_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        train_dataloader, uda_dataloader = dataloader.get_train_uda_dataloader(cfg)
        test_dataloader = dataloader.get_test_dataloader(cfg)

        bert.uda_bert(
            cfg,
            train_dataloader,
            uda_dataloader,
            test_dataloader,
        )