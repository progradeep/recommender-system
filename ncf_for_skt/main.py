import argparse
import os
from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader
from data_loader import get_infer_loader

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    if not os.path.exists(config.infer_path):
        os.makedirs(config.infer_path)

    if config.mode == 'train':
        num_users, num_items, train_loader, test_loader\
            = get_loader(data_path = config.data_path,
                        train_negs = config.train_negs,
                        test_negs = config.test_negs,
                        batch_size = config.batch_size,
                        num_workers = config.num_workers)
        solver = Solver(config, num_users, num_items)
        solver.train(train_loader, test_loader)
    elif config.mode == 'infer':
        num_users, num_items, infer_loader\
            = get_infer_loader(data_path = config.data_path,
                        train_negs = config.train_negs,
                        test_negs = config.test_negs,
                        batch_size = config.batch_size,
                        num_workers = config.num_workers)
        solver = Solver(config, num_users, num_items)
        solver.infer(infer_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('--model_type', type = str, default = 'gmf')

    # training hyper-parameters
    parser.add_argument('--latent_dim', type = int, default = 8)
    parser.add_argument('--lr', type = float, default = 0.01)
    parser.add_argument('--reg', type = float, default = 0)
    parser.add_argument('--num_epochs', type = int, default = 10)
    parser.add_argument('--batch_size', type = int, default = 200)
    parser.add_argument('--beta1', type = float, default = 0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type = float, default = 0.999)  # momentum2 in Adam
    parser.add_argument('--train_negs', type = int, default = 4)
    parser.add_argument('--test_negs', type = int, default = 99)

    # misc
    parser.add_argument('--mode', type = str, default = 'train')  # train or infer
    parser.add_argument('--num_workers', type = int, default = 2)
    parser.add_argument('--save_path', type = str, default = 'save')
    parser.add_argument('--infer_path', type = str, default = 'infer')
    parser.add_argument('--output_path', type = str, default = 'out.csv')
    parser.add_argument('--load_path', type = str, default = None)
    parser.add_argument('--data_path', type = str, default = '../../data/KISA_TBC_VIEWS_UNIQ_top50.csv')
    parser.add_argument('--log_step', type = int, default = 10000)
    parser.add_argument('--test_step', type = int, default = 1)
    parser.add_argument('--topk', type = int, default = 50)
    parser.add_argument('--use_gpu', type = bool, default = True)

    config = parser.parse_args()
    print(config)
    main(config)
