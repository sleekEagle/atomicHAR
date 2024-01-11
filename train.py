from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(cfg.utdmhad.path)
    train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(cfg)
    print('dataloaders obtained...')
    




if __name__ == "__main__":
    main()

