import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from test import *
from model import *
from tensorboard_logger import Logger
from thumos_features import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    utils.save_config(config, os.path.join(config.output_path, "config.txt"))

    net = Model(config.len_feature, config.num_classes, config.r_act, config.r_bkg)
    net = net.cuda()

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, supervision='weak',
                        seed=config.seed, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)
    print(type(test_loader))
    print(len(test_loader))
    test_info = {"step": [], "test_acc": [],
                "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
                "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [],
                "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []}
    # test_info = {"step": [], "test_acc": [], "average_mAP": [],
    #             "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
    #             "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []}
    
    logger = Logger(config.log_path)

    # test(net, config, logger, test_loader, test_info, 0, model_file=config.model_file)
    #
    # utils.save_best_record_thumos(test_info,
    #     os.path.join(config.output_path, "best_record.txt"))

    net.load_state_dict(torch.load(args.model_file))

    test(net, config, logger, test_loader, test_info, 0)

    utils.save_best_record_thumos(test_info,
                                  os.path.join(config.output_path, "best_record.txt"))