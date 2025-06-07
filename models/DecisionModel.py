
import torch
import torch.nn as nn
# from torchvision.ops.misc import Conv2dNormActivation
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet50
# from models.models import ConvNextModel, ResNetModel, VGGModel, MobileNetV2, DenseModel
from models.DecisionMobileViT import mobile_vit_small
import cv2
from torch import Tensor
from torchvision.ops import RoIAlign
from models.gcn import GCN



class Model(nn.Module):
    def __init__(self, backbone, nrois = 6, no_weight=False):
        super(Model, self).__init__()
        self.nrois = nrois #after rank
        # self.nrois_candidates = self.nrois*2
        self.no_weight = no_weight
        # if not self.no_weight:
        #     self.rpn = RPN(nrois=self.nrois_candidates).eval()
        # else:
        #     self.rpn = RPN(nrois=self.nrois).eval()
        roi_size = 7
        self.roi_align0 =  RoIAlign((roi_size,roi_size), 112./224, -1, aligned=True)
        # self.roi_align1 =  RoIAlign((roi_size,roi_size), 28./224, -1, aligned=True)
        # self.roi_align2 =  RoIAlign((roi_size,roi_size), 14./224, -1, aligned=True)
        # self.roi_align3 =  RoIAlign((roi_size,roi_size), 7./224, -1, aligned=True)

        # x_1 = self.roi_align(y['1'], rois_list, 28./224)
        # x_2 = self.roi_align(y['2'], rois_list, 14./224)
        # x_3 = self.roi_align(y['3'], rois_list, 7./224)
        # self.roi_align = RoIAlign((roi_size,roi_size), -1, aligned=True)
        self.local_pool = nn.AdaptiveMaxPool2d(1)
        # self.max_pool_1d = nn.AdaptiveMaxPool1d(1)
  
        self.self_state_len = 320
        self.gcn = GCN(nin=self.self_state_len, nhid=self.self_state_len, nout=self.self_state_len)

        # self.gcnl1, self.gcnl2 = GCN(nin=256, nhid=256, nout=256), GCN(nin=256, nhid=256, nout=256)

        self.action_predictor   = nn.Sequential(
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            nn.Linear(self.self_state_len, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.reason_predictor   = nn.Sequential(
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            nn.Linear(self.self_state_len, 64),
            nn.ReLU(),
            nn.Linear(64, 21)
        )
        # self.rnn.to('cuda') 
        # Step 2: Load Pre-Trained MaskRCNN
        # resnet_fpn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

        # Step 3: Extract the Backbone
        # self.backbone = resnet_fpn.backbone
        self.global_branch  = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # 全连接层，最后转换为[160, 1, 1]
            nn.Linear(8 * 7 * 7, 160),
            nn.ReLU()
        )
        self.global_branch = nn.AdaptiveMaxPool2d(1)

        if backbone == 'mobileViT':
            print('mobileViT backbone')
            self.backbone = mobile_vit_small()
            weights_dict = torch.load('models/mobilevit_s.pt', map_location='cpu')
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "classifier" in k:
                    del weights_dict[k]
            self.backbone.load_state_dict(weights_dict, strict=False)
        self.fpn = torchvision.ops.FeaturePyramidNetwork([32, 64, 96, 128, 160], 160)

    def select_topn_boxes(self, rois, weights, top_n):
        """
        Selects the top-k boxes based on the weights.

        :param rois: Tensor of shape [N, 20, 4] - the detected boxes.
        :param weights: Tensor of shape [N, 20] - the weights for each box.
        :param top_k: The number of top boxes to select.
        :return: Tuple of two tensors:
                - Tensor of shape [N, top_k, 4] containing the top-k boxes.
                - Tensor of shape [N, top_k] containing the corresponding weights.
        """


        # Check if there are less than top_k boxes
        n_boxes = rois.size(1)
        if n_boxes < top_n:
            raise ValueError(f"Number of boxes ({n_boxes}) is less than top_k ({top_k})")
        # Get the indices of the top_k weights
        top_weights, top_indices = torch.topk(weights, k=top_n, dim=1)
        top_indices = top_indices.to('cuda')
        # Gather the top-k boxes using the top_indices
        # print(rois.device)
        # print(top_indices.device)
        top_boxes = torch.gather(rois, 1, top_indices.unsqueeze(-1).expand(-1, -1, 4))
        return top_boxes, top_weights       
    def extract_values_from_boxes(self, gray_imgs, rpn_boxes):
        batch_size, _, H, W = gray_imgs.shape
        # tot_sums = torch.sum(gray_imgs)
        results = torch.zeros(batch_size, self.nrois_candidates)
        for i in range(batch_size):
            for j in range(self.nrois_candidates):
                box = rpn_boxes[i, j].long()  # Convert to long for indexing
                x1, y1, x2, y2 = box

                # Ensure the box is within the image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                # Extract the region and compute a value (e.g., mean)
                region = gray_imgs[i, 0, y1:y2+1, x1:x2+1]
                '''
                Notes: use mean here to avoid the big object to get the big attention
                '''
                value = region.mean()  # Change this as per your requirement

                results[i, j] = value
        return results
    


    def select_top_n_patches(self, images, n):
        batch_size, _, H, W = images.shape
        patch_size = 28
        patches_per_side = H // patch_size

        # Unfold the images to get the patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, 1, -1, patch_size * patch_size)

        # Calculate the mean intensity of each patch
        patch_intensities = patches.mean(dim=-1)

        # Select top n patches for each image in the batch
        _, top_patches_indices = torch.topk(patch_intensities, n, dim=-1)

        # Pre-create the tensor for storing the coordinates
        top_patches_coords = torch.zeros(batch_size, n, 4, dtype=torch.float)

        if self.no_weight:
            # Randomly generate patch indices
            total_patches = patches_per_side * patches_per_side
            for i in range(batch_size):
                random_indices = torch.randperm(total_patches)[:n]
                for j, idx in enumerate(random_indices):
                    row = idx // patches_per_side
                    col = idx % patches_per_side
                    x1, y1 = col * patch_size, row * patch_size
                    x2, y2 = x1 + patch_size, y1 + patch_size
                    top_patches_coords[i, j] = torch.tensor([x1, y1, x2, y2], dtype=torch.long)
            return top_patches_coords

        # Fill in the coordinates
        for i in range(batch_size):
            for j,  idx in enumerate(top_patches_indices[i, 0]):
                row = torch.div(idx, patches_per_side, rounding_mode='floor')
                col = idx % patches_per_side
                x1, y1 = col * patch_size, row * patch_size
                x2, y2 = x1 + patch_size, y1 + patch_size
                top_patches_coords[i, j] = torch.tensor([x1, y1, x2, y2], dtype=torch.long)
        return top_patches_coords
    def construct_graph(self, batch_size, nrois):
        intra_graph = torch.ones((batch_size, nrois, nrois))
        intra_graph = F.pad(intra_graph, (0, 1, 0, 1))
        intra_graph[:, nrois, nrois] = 1


        # inter_graph =  torch.zeros((batch_size, nrois+1, nrois+1))
        identity = torch.eye(nrois + 1)
        batch_identity = identity.unsqueeze(0).repeat(batch_size, 1, 1)

        inter_graph = batch_identity
        inter_graph[:, -1, :] = 1
        intra_graph =  intra_graph / intra_graph.sum(dim=-1, keepdim=True)
        inter_graph =  inter_graph / inter_graph.sum(dim=-1, keepdim=True)


        return intra_graph, inter_graph

        #return adj, there is total nrois + 1 node
    
    def gcn_forward(self, feats, intra_graph, inter_graph):
        feats = self.gcnl1(feats, intra_graph, inter_graph)
        feats = self.gcnl2(feats, intra_graph, inter_graph)
        return feats
                
    def forward(self, x, atten)->list[Tensor]:
        '''
        input:
        param x, [N, T, 3, H, W]
        param atten, [N, T, 1, H, W]
        '''
        # import pdb; pdb.set_trace()
        N, C, H, W = x.shape
        states =  torch.zeros(N, 1, self.self_state_len, device='cuda')
        # self.rpn.eval()
        # with torch.no_grad():
        #     rois = self.rpn(x.detach())#[N*T, 10, 4]
        #caculate the weights
        # atten = atten.view(N, 1,  H, W)
        rois  = self.select_top_n_patches(atten, self.nrois)
        rois = rois.cuda()
        # rois = select_top_n_patches()
        # if not self.no_weight:
        #     weights = self.extract_values_from_boxes(atten, rois) #[N, 20]
        #     rois, weights = self.select_topn_boxes(rois, weights, top_n = self.nrois)
        # weights = weights.view(N, self.nrois)
        rois_list = [r for r in rois]
        y = self.backbone(x)
        out_dict = {f"{i}": out for i, out in enumerate(y)}
        y = self.fpn(out_dict)
        global_feat = self.global_branch(y['4']) #[160, 7, 7]
        global_feat = global_feat.view(N, 1, -1)
        global_feat = global_feat.expand(N, self.nrois, -1)
        x_0 = self.roi_align0(y['0'], rois_list)
        # x_1 = self.roi_align1(y['1'], rois_list)
        # x_2 = self.roi_align2(y['2'], rois_list)
        # x_3 = self.roi_align3(y['3'], rois_list)
        # x_list = [x_0, x_1, x_2, x_3]
        # y = torch.cat(x_list, dim=1)
        # x_0 = self.adapter(x_0)
        y = x_0

        # y = self.roi_align(y, rois_list)
        y = self.local_pool(y).squeeze_(-1).squeeze(-1)
        y = y.view(N, self.nrois, -1) #N, nrois, c
        y = torch.cat([y, global_feat], dim=-1)
        feats = torch.cat((y, states), dim=1)
        intra_graph, inter_graph = self.construct_graph(N, self.nrois) #return adj, there is total nrois + 1 node
        intra_graph, inter_graph = intra_graph.cuda(), inter_graph.cuda()
        feats = self.gcn(feats,  inter_graph) # [N, nrois+1, c]
        states_output = torch.flatten(feats[:, self.nrois, :], start_dim=1)
        action_logits = self.action_predictor(states_output)
        reason_logits = self.reason_predictor(states_output)

      
        return action_logits, reason_logits
      
