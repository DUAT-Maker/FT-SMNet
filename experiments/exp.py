import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
from data_process.data_loader import Dataset_Custom
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.metrics import metric
from model.FT_SMNet import FT_SMNet


class Exp(object):
    def __init__(self, args):
        self.args = args
        self.args.devices = args.devices
        self.model = self._build_model().to(self.args.devices)

    def _build_model(self):           
        model = FT_SMNet(self.args.seq_len, self.args.pred_len, self.args.Features, self.args.k_size)
        print(model)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter=========: %.2fM" % (total/1e6))
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'FuXing': Dataset_Custom,
            'FuXing_MAV': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
    
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; 
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            inverse=args.inverse,
            empty_ratio = args.empty_ratio
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []
        pred_MSE = []
        pred_MAE = []
        pred_scale_MSE = []
        pred_scale_MAE = []
       
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            pred, pred_scale, true, true_scale = self._process_one_batch_FT_SMNet(
                valid_data, batch_x, batch_y)
            pred, pred_scale, true, true_scale = pred[:,:,:-1], pred_scale[:,:,:-1], true[:,:,:-1], true_scale[:,:,:-1]
            loss = criterion(pred, true).detach().cpu().numpy()
            pred_MSE.append(nn.MSELoss()(pred, true).detach().cpu().numpy())
            pred_scale_MSE.append(nn.MSELoss()(pred_scale, true_scale).detach().cpu().numpy())
            pred_MAE.append(nn.L1Loss()(pred, true).detach().cpu().numpy())
            pred_scale_MAE.append(nn.L1Loss()(pred_scale, true_scale).detach().cpu().numpy())
            total_loss.append(loss)
        total_loss = np.average(total_loss)

        print('normed mse:{:.4f}, mae:{:.4f}'.format(np.mean(pred_MSE), np.mean(pred_MAE)))
        print('denormed mse:{:.4f}, mae:{:.4f}'.format(np.mean(pred_scale_MSE), np.mean(pred_scale_MAE)))
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)
        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred, pred_scale, true, true_scale = self._process_one_batch_FT_SMNet(
                    train_data, batch_x, batch_y)

                # Drop the last dimensions: temperature
                pred, pred_scale, true, true_scale = pred[:, :, :-1], pred_scale[:, :, :-1], true[:, :, :-1], true_scale[:, :, :-1]
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward(retain_graph=True)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))
            
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)       
        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds = []
        trues = []  
        pred_scales = []
        true_scales = []
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        for i, (batch_x,batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, true, true_scale = self._process_one_batch_FT_SMNet(
                test_data, batch_x, batch_y)
            preds.append(np.concatenate((pred.detach().cpu().numpy(),batch_y_mark[:, :, np.newaxis]),axis=2))
            trues.append(np.concatenate((true.detach().cpu().numpy(),batch_y_mark[:, :, np.newaxis]),axis=2))
            pred_scales.append(np.concatenate((pred_scale.detach().cpu().numpy(),batch_y_mark[:, :, np.newaxis]),axis=2))
            true_scales.append(np.concatenate((true_scale.detach().cpu().numpy(),batch_y_mark[:, :, np.newaxis]),axis=2))

        preds = np.array(preds)
        trues = np.array(trues)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

        # Drop the last 2 dimensions: temperature and date
        mae, mse, rmse, mape, mspe, corr = metric(preds[:,:,:-2], trues[:,:,:-2])
        maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales[:,:,:-2], true_scales[:,:,:-2])
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
        print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
      
        if self.args.save:
            folder_path = self.args.folder_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            mae, mse, rmse, mape, mspe, corr = metric(preds[:,:,:-2], trues[:,:,:-2])
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred_scales.npy', pred_scales)
            np.save(folder_path + 'true_scales.npy', true_scales)
        return mae, maes, mse, mses

    def _process_one_batch_FT_SMNet(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.args.devices)
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        outputs_scaled = dataset_object.inverse_transform(outputs)
        batch_y = batch_y[:,-self.args.pred_len:,:].to(self.args.devices)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)
        return outputs, outputs_scaled, batch_y, batch_y_scaled
       
