import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset_two_stream import restore_full_sequence
from dataset_two_stream import get_data_dict
from dataset_two_stream import VideoFeatureDataset
# from model_former1 import ASDiffusionModel_RGB
from model_two_stream import ASDiffusionModel
# from model_two_stream import ASDiffusionModel_flow
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


class Trainer:
    def __init__(self, input_dim_RGB, input_dim_FLOW, encoder_params_RGB, encoder_params_FLOW, decoder_params_RGB, decoder_params_FLOW, diffusion_params_RGB,  diffusion_params_FLOW, event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
#         self.encoder_params = encoder_params
#         self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess
        self.RGB_model = ASDiffusionModel(input_dim_RGB, encoder_params_RGB, decoder_params_RGB, diffusion_params_RGB, self.num_classes, self.device)
        
        self.FLOW_model = ASDiffusionModel(input_dim_FLOW, encoder_params_FLOW, decoder_params_FLOW, diffusion_params_FLOW, self.num_classes, self.device)
        print('RGB Model Size: ', sum(p.numel() for p in self.RGB_model.parameters()))
        print('FLOW Model Size: ', sum(p.numel() for p in self.FLOW_model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.RGB_model.to(device)
        self.FLOW_model.to(device)

        rgb_optimizer = optim.Adam(self.RGB_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        flow_optimizer = optim.Adam(self.FLOW_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        rgb_optimizer.zero_grad()
        flow_optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.RGB_model.load_state_dict(saved_state['rgb_model'])
                    self.FLOW_model.load_state_dict(saved_state['flow_model'])
                    rgb_optimizer.load_state_dict(saved_state['rgb_optimizer'])
                    flow_optimizer.load_state_dict(saved_state['flow_optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=0)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.RGB_model.train()
            self.FLOW_model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                feature_rgb, feature_flow, label, boundary, video = data
                feature_rgb, feature_flow, label, boundary = feature_rgb.to(device), feature_flow.to(device), label.to(device), boundary.to(device)
                event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1)
                rgb_loss_dict = self.RGB_model.get_training_loss(feature_rgb,
                    event_gt,
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                flow_loss_dict = self.FLOW_model.get_training_loss(feature_flow,
                     event_gt,
                     boundary_gt=boundary,
                     encoder_ce_criterion=ce_criterion,
                     encoder_mse_criterion=mse_criterion,
                     encoder_boundary_criterion=bce_criterion,
                     decoder_ce_criterion=ce_criterion,
                     decoder_mse_criterion=mse_criterion,
                     decoder_boundary_criterion=bce_criterion,
                     soft_label=soft_label
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in rgb_loss_dict.items():
                    total_loss += loss_weights[k] * v

                for k,v in flow_loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in rgb_loss_dict.items():
                        logger.add_scalar(f'Train-RGB-{k}', loss_weights[k] * v.item() / batch_size, step)
                    for k,v in flow_loss_dict.items():
                        logger.add_scalar(f'Train-flow-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    rgb_optimizer.step()
                    flow_optimizer.step()
                    rgb_optimizer.zero_grad()
                    flow_optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if result_dir:

                state = {
                    'rgb_model': self.RGB_model.state_dict(),
                    'flow_model': self.FLOW_model.state_dict(),
                    'rgb_optimizer': rgb_optimizer.state_dict(),
                    'flow_optimizer': flow_optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.RGB_model.state_dict(), f'{result_dir}/rgb_epoch-{epoch}.model')
                    torch.save(self.FLOW_model.state_dict(), f'{result_dir}/flow_epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k,v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k,v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k,v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k,v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()
            
    def load_and_test(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label, num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):
        
        device = self.device
        self.RGB_model.eval()
        self.FLOW_model.eval()
        self.RGB_model.to(device)
        self.FLOW_model.to(device)

        for epoch in range(num_epochs):
            epoch += 670
            if result_dir and epoch%5 == 0:
                rgb_model_path = f'{result_dir}/rgb_epoch-{epoch}.model'
                flow_model_path = f'{result_dir}/flow_epoch-{epoch}.model'
                self.RGB_model.load_state_dict(torch.load(rgb_model_path, map_location=device))
                self.FLOW_model.load_state_dict(torch.load(flow_model_path, map_location=device))

                if result_dir:
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    logger = SummaryWriter(result_dir)
                    for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                        test_result_dict = self.test(
                            test_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k,v in test_result_dict.items():
                                logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                            np.save(os.path.join(result_dir, 
                                f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                        for k,v in test_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                        if log_train_results:

                            train_result_dict = self.test(
                                train_test_dataset, mode, device, label_dir,
                                result_dir=result_dir, model_path=None)

                            if result_dir:
                                for k,v in train_result_dict.items():
                                    logger.add_scalar(f'Train-{mode}-{k}', v, epoch)

                                np.save(os.path.join(result_dir, 
                                    f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)

                            for k,v in train_result_dict.items():
                                print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        

        return result_dict


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])


        self.RGB_model.eval()
        self.FLOW_model.eval()
        self.RGB_model.to(device)
        self.FLOW_model.to(device)

        if model_path:
            self.RGB_model.load_state_dict(torch.load(model_path + '_RGB'))
            self.FLOW_model.load_state_dict(torch.load(model_path + '_FLOW'))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature_RGB, feature_FLOW, label, _, video = test_dataset[video_idx]
            
            torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output_RGB = [self.RGB_model.encoder(feature_RGB[i].to(device)) 
                       for i in range(len(feature_RGB))] # output is a list of tuples
                output_FLOW = [self.FLOW_model.encoder(feature_FLOW[i].to(device)) for i in range(len(feature_FLOW))]
                output_RGB = [torch.softmax(i, 1).cpu() for i in output_RGB]
                output_FLOW = [torch.softmax(i, 1).cpu() for i in output_FLOW]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output_RGB = [self.RGB_model.ddim_sample(feature_RGB[i].to(device), seed) for i in range(len(feature_RGB))]
                output_FLOW = [self.FLOW_model.ddim_sample(feature_FLOW[i].to(device), seed) for i in range(len(feature_FLOW))]
                output_RGB = [i.cpu() for i in output_RGB]
                output_FLOW = [i.cpu() for i in output_FLOW]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output_RGB = [self.RGB_model.ddim_sample(feature_RGB[len(feature_RGB)//2].to(device), seed)]
                output_FLOW = [self.FLOW_model.ddim_sample(feature_FLOW[len(feature_FLOW)//2].to(device), seed)]
                output_RGB = [i.cpu() for i in output_RGB]
                output_FLOW = [i.cpu() for i in output_FLOW]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output_RGB[0].shape[0] == 1)
            assert(output_FLOW[0].shape[0] == 1)

            # --- 新增：统计结束并计算 ---
            end_event.record()
            torch.cuda.synchronize() # 强制同步，确保GPU计算完成
            
            elapsed_time_ms = start_event.elapsed_time(end_event)
            seconds = elapsed_time_ms / 1000.0
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # 转换为 GB
            
            # 计算总帧数（label的最后一个维度）
            total_frames = label.shape[-1]
            fps = total_frames / seconds
            
            print(f"\n[Video: {video}]")
            print(f"Inference Time: {seconds:.4f}s | Speed: {fps:.2f} FPS")
            print(f"Peak GPU Memory: {peak_mem:.2f} GB")
            # ---------------------------
            
            
            min_len = min([i.shape[2] for i in output_FLOW])
            output_RGB = [i[:,:,:min_len] for i in output_RGB]
            output_FLOW = [i[:,:,:min_len] for i in output_FLOW]
            output_RGB = torch.cat(output_RGB, 0)  # torch.Size([sample_rate, C, T])
            output_FLOW = torch.cat(output_FLOW, 0)  # torch.Size([sample_rate, C, T])
            output_RGB = output_RGB.mean(0).numpy()
            output_FLOW = output_FLOW.mean(0).numpy()
            
            output = 0.7*output_RGB + 0.3* output_FLOW
 
            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            # ========== 新增：在这里保存 argmax 之前的概率矩阵为 .npy 文件 ==========
            save_dir = 'output_npy_fusion'  # 保存的文件夹名
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 保存为 .npy 格式
            save_path = os.path.join(save_dir, f'{video}_fusion.npy')
            np.save(save_path, output)
            # ======================================================================

            output = np.argmax(output, 0)

            output = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()

            assert(output.shape == label.shape)
            
            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.RGB_model.eval()
        self.FLOW_model.eval()
        self.RGB_model.to(device)
        self.FLOW_model.to(device)

        if model_path:
            self.model_RGB.load_state_dict(torch.load(model_path + '_RGB'))
            self.model_FLOW.load_state_dict(torch.load(model_path + '_FLOW'))
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]
                
                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        acc, edit, f1s = func_eval(
            label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    feature_dir_768 = os.path.join(root_data_dir, dataset_name, 'features_768')
    feature_dir_FLOW = os.path.join(root_data_dir, dataset_name, 'flow_features')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)
    input_dim_RGB = 768
    input_dim_FLOW = 1024
    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    train_data_dict = get_data_dict(
        feature_dir_RGB=feature_dir_768,
        feature_dir_FLOW=feature_dir_FLOW,
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_data_dict = get_data_dict(
        feature_dir_RGB=feature_dir_768,
        feature_dir_FLOW=feature_dir_FLOW,
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')
    
    trainer = Trainer(input_dim_RGB, input_dim_FLOW, dict(encoder_params), dict(encoder_params), dict(decoder_params), dict(decoder_params), dict(diffusion_params), dict(diffusion_params), event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.load_and_test(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
        log_freq=log_freq, log_train_results=log_train_results
    )
