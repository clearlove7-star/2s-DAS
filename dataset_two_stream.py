import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d

def get_data_dict(feature_dir_RGB, feature_dir_FLOW, label_dir, video_list, event_list, sample_rate=4, temporal_aug=True, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature_rgb': None,
        'feature_flow': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file_RGB = os.path.join(feature_dir_RGB, '{}.npy'.format(video))
        feature_file_FLOW = os.path.join(feature_dir_FLOW, '{}.npy'.format(video))
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        feature_rgb = np.load(feature_file_RGB, allow_pickle=True).T
        feature_flow = np.load(feature_file_FLOW, allow_pickle=True)
        if(feature_rgb.shape[1]>feature_flow.shape[1]):
            dim = feature_rgb.shape[1] - feature_flow.shape[1]
            feature_rgb = feature_rgb[:,:-dim]
        else:
            dim = feature_flow.shape[1] - feature_rgb.shape[1]
            padding_zeros = np.zeros((feature_rgb.shape[0], dim))
            feature_rgb = np.concatenate((feature_rgb, padding_zeros), axis=1)
                    
        if len(feature_rgb.shape) == 3:
            feature_rgb = np.swapaxes(feature_rgb, 0, 1)
        elif len(feature_rgb.shape) == 2:
            feature_rgb = np.swapaxes(feature_rgb, 0, 1)
            feature_rgb = np.expand_dims(feature_rgb, 0)
        else:
            raise Exception('Invalid RGB Feature.')

        if len(feature_flow.shape) == 3:
            feature_flow = np.swapaxes(feature_flow, 0, 1)
        elif len(feature_flow.shape) == 2:
            feature_flow = np.swapaxes(feature_flow, 0, 1)
            feature_flow = np.expand_dims(feature_flow, 0)
        else:
            raise Exception('Invalid Flow Feature.')

        assert (feature_rgb.shape[1] == event_seq_raw.shape[0])
        assert (feature_rgb.shape[1] == boundary_seq_raw.shape[0])
        assert (feature_flow.shape[1] == event_seq_raw.shape[0])
        assert (feature_flow.shape[1] == boundary_seq_raw.shape[0])
                                
        if temporal_aug:

            feature_rgb = [
                feature_rgb[:, offset::sample_rate, :]
                for offset in range(sample_rate)
            ]

            feature_flow = [
                feature_flow[:, offset::sample_rate, :]
                for offset in range(sample_rate)
            ]

            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        else:
            feature_rgb = [feature_rgb[:,::sample_rate,:]]
            feature_flow = [feature_flow[:,::sample_rate,:]]
            event_seq_ext = [event_seq_raw[::sample_rate]]
            boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

        data_dict[video]['feature_rgb'] = [torch.from_numpy(i).float() for i in feature_rgb]
        data_dict[video]['feature_flow'] = [torch.from_numpy(i).float() for i in feature_flow]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['event_seq_ext'] = [torch.from_numpy(i).float() for i in event_seq_ext]
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()
        data_dict[video]['boundary_seq_ext'] = [torch.from_numpy(i).float() for i in boundary_seq_ext]
        
    return data_dict

def get_boundary_seq(event_seq, boundary_smooth=None):

    boundary_seq = np.zeros_like(event_seq)

    _, start_times, end_times = get_labels_start_end_time([str(int(i)) for i in event_seq])
    boundaries = start_times[1:]
    assert min(boundaries) > 0
    boundary_seq[boundaries] = 1
    boundary_seq[[i-1 for i in boundaries]] = 1

    if boundary_smooth is not None:
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
        
        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        temp_seq[temp_seq.shape[0] // 2] = 1
        temp_seq[temp_seq.shape[0] // 2 - 1] = 1
        norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        boundary_seq /= boundary_seq.max()

    return boundary_seq


def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
        
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out




class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, class_num, mode):
        super(VideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]
        
    def get_class_weights(self):
        
        full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)

        return class_weights

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature_rgb = self.data_dict[video]['feature_rgb']
            feature_flow = self.data_dict[video]['feature_flow']
            label = self.data_dict[video]['event_seq_ext']
            boundary = self.data_dict[video]['boundary_seq_ext']

            temporal_aug_num = len(feature_flow)
            temporal_rid = random.randint(0, temporal_aug_num - 1) # a<=x<=b
            feature_rgb = feature_rgb[temporal_rid]
            feature_flow = feature_flow[temporal_rid]
            label = label[temporal_rid]
            boundary = boundary[temporal_rid]

            spatial_aug_num = feature_flow.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1) # a<=x<=b
            feature_rgb = feature_rgb[spatial_rid]
            feature_flow = feature_flow[spatial_rid]
            
            feature_rgb = feature_rgb.T   # F x T
            feature_flow = feature_flow.T

            boundary = boundary.unsqueeze(0)
            boundary /= boundary.max()  # normalize again
            
        if self.mode == 'test':

            feature_rgb = self.data_dict[video]['feature_rgb']
            feature_flow = self.data_dict[video]['feature_flow']
            label = self.data_dict[video]['event_seq_raw']
            boundary = self.data_dict[video]['boundary_seq_ext']  # boundary_seq_raw not used

            feature_rgb = [torch.swapaxes(i, 1, 2) for i in feature_rgb]  # [10 x F x T]
            feature_flow = [torch.swapaxes(i, 1, 2) for i in feature_flow]  # [10 x F x T]
            label = label.unsqueeze(0)   # 1 X T'  
            boundary = [i.unsqueeze(0).unsqueeze(0) for i in boundary]   # [1 x 1 x T]  

        return feature_rgb, feature_flow, label, boundary, video

    
