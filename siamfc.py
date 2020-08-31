from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR

from got10k.trackers import Tracker


class SiamFC(nn.Module):

    def __init__(self,features=256,  M=3, G=8, r=2, L=32):
        super(SiamFC, self).__init__()


        # conv1
        self.layer1 =nn.Sequential(
                nn.Conv2d(3, 96, 11, 2),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2))
        # conv2
        self.layer2 =nn.Sequential(
                nn.Conv2d(96, 256, 5, 1, groups=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2))
        # conv3
        self.layer3 =nn.Sequential(
                nn.Conv2d(256, 384, 3, 1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True))
        # conv4
        self.layer4 =nn.Sequential(
                nn.Conv2d(384, 384, 3, 1, groups=2),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True))
        # conv5
        self.layer5 =nn.Sequential(
                nn.Conv2d(384, 256, 3, 1, groups=2))
        
        self.MaxPool =nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2))
        
      
        self.Conv1_1 =nn.Sequential(
                nn.Conv2d(96, 256, 9, 1, groups=2),
                nn.BatchNorm2d(256))
        
        self.Conv3_1 =nn.Sequential(
                nn.Conv2d(384, 256, 5, 1, groups=2),
                nn.BatchNorm2d(256))	
			
        self.conv_se = nn.Sequential(
                nn.Conv2d(features, features//r, 1, padding=0, bias=True),
                nn.ReLU(inplace=True)
        )
       
        self.conv_ex = nn.Sequential(nn.Conv2d(features//r, features, 1, padding=0, bias=True))
        self.gap_1 = nn.AdaptiveAvgPool2d(1)

       
        # adjust layer as in the original SiamFC in matconvnet
        self.adjust = nn.Conv2d(1, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()


        self._initialize_weights()
   
    def learn_z_cat(self, template):
        
        z1 = self.layer1(template) 
        z2 = self.layer2(z1) 
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        
        z5 = self.layer5(z4)
        z11_pool= self.MaxPool(z1)
        z1_1 = self.Conv1_1(z11_pool)
        z3_1 = self.Conv3_1(z3)
        n, c, h, w = z5.size()
        
        
        return z1_1, z3_1 , z5
        
    def learn_att_z(self, z1_1, z3_1 , z5):
        features = torch.cat([z1_1, z3_1 , z5], dim=1)

        feas = torch.cat([z1_1.clone().unsqueeze(dim=1), z3_1.clone().unsqueeze(dim=1), z5.clone().unsqueeze(dim=1)], dim=1)
       
        U = torch.sum(feas.clone(), dim=1)
        S = self.gap_1(U)
        Z = self.conv_se(S)
        fz1=self.conv_ex(Z).unsqueeze(dim=1)
        fz2=self.conv_ex(Z).unsqueeze(dim=1)
        fz3=self.conv_ex(Z).unsqueeze(dim=1)
        attention_vector = torch.cat([fz1, fz2, fz3], dim=1)
        attention_vector = self.sigmoid(attention_vector.clone())

        #V = (feas.clone() * attention_vector.clone()).sum(dim=1)
        V1=z1_1.clone() * attention_vector[:,0,:,:,:].clone()
        V2=z3_1.clone() * attention_vector[:,1,:,:,:].clone()
        V3=z5.clone() * attention_vector[:,2,:,:,:].clone()
        V_f = torch.cat([V1, V2, V3], dim=1)

        V = features+V_f
        
        return V
    
    def learn_x(self, instanse):
        x1 = self.layer1(instanse) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x11_pool= self.MaxPool(x1)
        x1_1 = self.Conv1_1(x11_pool)
        x3_1 = self.Conv3_1(x3)
        features = torch.cat([x1_1, x3_1 , x5], dim=1)
        return features


    def forward(self, z, x):

 
        z1_1, z3_1 , z5 = self.learn_z_cat(z)
        z_feat = self.learn_att_z(z1_1, z3_1 , z5)
        x_feat = self.learn_x(x)

        # fast cross correlation
        out = self.fast_cross(x_feat, z_feat, "out", 0.1)
        

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def fast_cross(self, x, z, img_name = None, num = 0.001):
        n, c, h, w = x.size()
        out = F.conv2d(x.view(1, n * c, h, w), z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        if img_name != None:
            out_img = num * out + 0.0
            out_img = np.asarray(out_img[0].permute(1,2,0).detach().cpu())
            cv2.imwrite("{}.png".format(img_name), out_img)

        return out



class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(
            name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 135,
            'instance_sz': 263,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            z1_1, z3_1 , z5 = self.net.learn_z_cat(exemplar_image)
            self.kernel = self.net.learn_att_z(z1_1, z3_1 , z5)

            #self.kernel = self.net.learn_z(exemplar_image)
            

    def update(self, image):
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()

            instances = self.net.learn_x(instance_images)
            responses = F.conv2d(instances, self.kernel) * 0.001
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            self.net.train()
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval()

        z = batch[0].to(self.device)
        #z_noise = batch[1].to(self.device)

        x = batch[1].to(self.device)
        #x_noise = batch[3].to(self.device)

        with torch.set_grad_enabled(backward):
            responses = self.net(z, x)
            labels, weights = self._create_labels(responses.size())
            loss = F.binary_cross_entropy_with_logits(
                responses, labels, weight=weights, size_average=True)

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        #print(loss.item())
        return loss.item()

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights
