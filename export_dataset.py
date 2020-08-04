from utils import common, configuration, dataloader
import bert

if __name__ == "__main__":

    cfg_json_list = [
        # "config/uda/imdb_20_uda_backtranslation.json",
        "config/uda/imdb_20_uda_sr.json",
        ]

    for cfg_json in cfg_json_list:

        cfg = configuration.uda_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        dataloader.export_uda_dataset(cfg)