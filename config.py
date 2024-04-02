import argparse
import time
import torch



def arg_parse():
    parser = argparse.ArgumentParser()




    # parser.add_argument("--mode", default = 'train', help = "Setting the mode type. (train / explain)")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', help="Setting the task type. (train / explain)")
    # gnn模型“训练”阶段的参数设置
    parser.add_argument("--dataset", default = 'BBBP', help = "Set the datasets. (BA_shapes / )")
    # parser.add_argument("--epoch", default = 5000, help = "Epoch, in training stage. (A number such as 100)")
    # parser.add_argument("--batch_size", default = 64, help = "Batch size, in training stage. (A number such as 32)")
    # parser.add_argument("--lr", default = 0.01, help = "Learn rate. (A number such as 0.001)")


    # gnn模型“解释”阶段的参数设置
    parser.add_argument("--epoch_E", default = 11, help="Epoch, in explanation stage. (A number such as 100)")
    parser.add_argument("--batch_size_E", default=128, help="batch_size, in explanation stage. (A number such as 32)")
    parser.add_argument("--lr_E", default=0.001, help="Learn rate, in explanation stage. (A number such as 0.001)")

    return parser.parse_args()




if __name__ == '__main__':

    args = arg_parse()
