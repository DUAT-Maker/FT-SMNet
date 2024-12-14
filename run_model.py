import argparse
import os
import torch
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from experiments.exp import Exp

parser = argparse.ArgumentParser(description='FuXing dataset')
parser.add_argument('--model_name', type=str, required=False, default='FT-SMNet', help='model of the experiment')

### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='FuXing', choices=['FuXing','FuXing_MAV'], help='name of dataset')
parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='FuXing.csv', help='location of the data file')
parser.add_argument('--checkpoints', type=str, default='exp/checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')

### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--devices', type=str, default='cuda:0',help='device ids of multile gpus')
                                                                                  
### -------  training settings --------------                 
parser.add_argument('--seq_len', type=int, default=539, help='input sequence length of SCINet encoder, look back window')
parser.add_argument('--label_len', type=int, default=0 , help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=77, help='prediction sequence length, horizon')
parser.add_argument('--Features', default=10, type=int)                  
parser.add_argument('--k_size', default=2, type=int)
parser.add_argument('--empty_ratio', default=0, type=float, help='Randomly set some data to zero')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--save', type=bool, default =True, help='save the output results')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

if args.evaluate:
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}'.format(args.model_name, args.data, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size, )
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
else:
    setting = '{}_{}_sl{}_ll{}_pl{}_lr{}_bs{}'.format(args.model_name, args.data, args.seq_len, args.label_len, args.pred_len, args.lr, args.batch_size)
    args.folder_path = 'exp/'+'results_empty_ratio_'+str(args.empty_ratio)+ '/' #save results
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting)
    with open(args.folder_path+'setting.txt','w') as f:
        f.write('model_name:{}\n'.format(args.model_name))
        f.write('data set:{}\n'.format(args.data))
        f.write('sequence length:{}\n'.format(args.seq_len))
        f.write('label length:{}\n'.format(args.label_len))
        f.write('pred length:{}\n'.format(args.pred_len))
        f.write('train_epochs:{}\n'.format(args.train_epochs))
        f.write('batch_size:{}\n'.format(args.batch_size))
        f.write('patience:{}\n'.format(args.patience))
        f.write('lr:{}\n'.format(args.lr))
        f.write('loss:{}\n'.format(args.loss))
        f.write('folder_path:{}\n'.format(args.folder_path))


