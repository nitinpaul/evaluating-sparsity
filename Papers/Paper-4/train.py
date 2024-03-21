import init_path
import utils.distributed as dist
import utils.trainer as trainer
import config
from config import cfg


def main():
    config.load_cfg_fom_args("Train a CFCD model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)


if __name__ == "__main__":
    main()
  
