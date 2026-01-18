class Path_Hyperparameter:
    random_seed = 42

    # dataset hyper-parameter
    dataset_name = '/T2007061/lrz_workspace/dataset/LEVIR-CD256'
    # dataset_name = '/T2007061/lrz_workspace/SYSU-CD256'
    # dataset_name = '/T2007061/lrz_workspace/dataset/ChangeDetectionDataset/Real/subset'
    # dataset_name = '/T2007061/lrz_workspace/dataset/WHU-CD-256'
    
    
    # training hyper-parameter
    epochs: int = 200  # Number of epochs
    batch_size: int = 8  # Batch size
    inference_ratio = 2  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 2e-4  # Learning rate
    factor = 0.1  # learning rate decreasing factor
    patience = 12  # schedular patience
    warm_up_step = 500  # warm up step
    weight_decay: float = 1e-3  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    #load: str = '/T2007061/lrz_workspace/FDAM-NET/weight/base_WHU_CD_1/WHU_CD256_best_f1score_model/best_f1score_epoch361_Wed Oct  8 23:56:54 2025.pth'
    
    #load: str = False  # Load model and/or optimizer from a .pth file for testing or continuing training
    max_norm: float = 20  # gradient clip max norm

    # evaluate hyper-parameter
    #evaluate_epoch: int = 250  # start evaluate after training for evaluate epochs
    evaluate_epoch: int = 100
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = True  # if save checkpoint of model or not
    save_interval: int = 10  # save checkpoint every interval epoch
    save_best_model: bool = True  # if save best model or not

    # log wandb hyper-parameter
    log_wandb_project: str = 'creloss'  # wandb project name

    # data transform hyper-parameter
    noise_p: float = 0.3  # probability of adding noise

    # model hyper-parameter
    dropout_p: float = 0.1  # probability of dropout
    patch_size: int = 256  # size of input image

    y = 2  # ECA-net parameter
    b = 1  # ECA-net parameter

    # inference parameter
    log_path = './log_feature/'

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}


ph = Path_Hyperparameter()
