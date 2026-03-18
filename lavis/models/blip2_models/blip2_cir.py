"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import random
from math import sqrt
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from torchvision.transforms import ToPILImage
import os
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from sklearn.cluster import KMeans



def l2norm(X, dim=-1):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def l1norm(X, dim):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return 

def info_nce(query, target):
    bs = query.size(0)
    targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(query.device)
    temp = nn.Parameter(0.07 * torch.ones([]))
    x = torch.matmul(query,target).squeeze().to(query.device)
    #print('x',x.shape)
    sim_i2t,_ = x.max(-1)
    sim_i2t = sim_i2t / temp
    return F.cross_entropy(sim_i2t, targets)

def colorful_spectrum_mix(img1, img2, alpha=1.0, ratio=1.0):
    """
    Pure PyTorch FFT mixup on frequency amplitude.
    Inputs:
        img1, img2: torch.Tensor, shape [B, C, H, W], value in [0, 1] float
        alpha: scalar float
        ratio: float in (0, 1], determines crop size in frequency domain
    Returns:
        mixed1, mixed2: torch.Tensor, shape [B, C, H, W], value in [0, 1]
    """
    assert img1.shape == img2.shape and img1.dim() == 4  # [B, C, H, W]
    B, C, H, W = img1.shape
    device = img1.device

    h_crop = int(H * sqrt(ratio))
    w_crop = int(W * sqrt(ratio))
    h_start = H // 2 - h_crop // 2
    w_start = W // 2 - w_crop // 2

    # Fourier transform
    img1_fft = torch.fft.fft2(img1, dim=(-2, -1))
    img2_fft = torch.fft.fft2(img2, dim=(-2, -1))

    img1_abs, img1_phase = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_phase = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_abs = torch.fft.fftshift(img1_abs, dim=(-2, -1))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(-2, -1))

    img1_abs_ = img1_abs.clone()
    img2_abs_ = img2_abs.clone()

    for i in range(B):
        lam = torch.empty(1).uniform_(0, alpha).item()
        img1_abs[i, :, h_start:h_start+h_crop, w_start:w_start+w_crop] = \
            lam * img2_abs_[i, :, h_start:h_start+h_crop, w_start:w_start+w_crop] + \
            (1 - lam) * img1_abs_[i, :, h_start:h_start+h_crop, w_start:w_start+w_crop]

        img2_abs[i, :, h_start:h_start+h_crop, w_start:w_start+w_crop] = \
            lam * img1_abs_[i, :, h_start:h_start+h_crop, w_start:w_start+w_crop] + \
            (1 - lam) * img2_abs_[i, :, h_start:h_start+h_crop, w_start:w_start+w_crop]

    img1_abs = torch.fft.ifftshift(img1_abs, dim=(-2, -1))
    img2_abs = torch.fft.ifftshift(img2_abs, dim=(-2, -1))

    mixed1_fft = img1_abs * torch.exp(1j * img1_phase)
    mixed2_fft = img2_abs * torch.exp(1j * img2_phase)

    mixed1 = torch.fft.ifft2(mixed1_fft, dim=(-2, -1)).real
    mixed2 = torch.fft.ifft2(mixed2_fft, dim=(-2, -1)).real

    # Clip to [0, 1]
    mixed1 = mixed1.clamp(0, 1)
    mixed2 = mixed2.clamp(0, 1)

    return mixed1, mixed2


@registry.register_model("Blip2QformerCir")
class Blip2QformerCir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()
        print("Loading model...")
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj_cau = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.loss_T = nn.Parameter(torch.tensor([10.]))
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        W = torch.Tensor(embed_dim, embed_dim)
        self.W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: embed_dim].to(self.device)
        norm = torch.norm(self.W, p=2, dim=1, keepdim=True)
        self.W = self.W / norm

        self.conW = self.W
        
        # self.style_intervener = StyleIntervener(temperature=0.07)
        
    def semantic_consistency_confidence(self,
                                            fused_feature,
                                            target_feature,
                                            temperature=0.1,
                                            uncertainty_factor_weight=0.3,
                                            feature_sim_weight=0.4,
                                            pred_sim_weight=0.3):

        # 特征空间相似度
        fused_norm = F.normalize(fused_feature, p=2, dim=1)
        target_norm = F.normalize(target_feature, p=2, dim=1)
        feature_sim = torch.sum(fused_norm * target_norm, dim=1)
        self.conW = self.conW.to(fused_feature.device)
        # 预测空间相似度
        fused_pred = fused_feature.mm(self.conW)
        target_pred = target_feature.mm(self.conW)

        # 计算JS散度作为分布差异度量
        fused_prob = F.softmax(fused_pred / temperature, dim=1)
        target_prob = F.softmax(target_pred / temperature, dim=1)

        m = 0.5 * (fused_prob + target_prob)
        js_div = 0.5 * (F.kl_div(m.log(), fused_prob, reduction='none').sum(dim=1)) + 0.5 * (F.kl_div(m.log(), target_prob, reduction='none').sum(dim=1))

        # 将散度转换为相似度 (0-1范围)
        pred_sim = 1 - torch.tanh(js_div)

        # 不确定性评估
        # 计算每个特征的预测确定性
        fused_max_prob = fused_prob.max(dim=1)[0]
        target_max_prob = target_prob.max(dim=1)[0]

        # 综合确定性因子 (两者都确定时值高)
        certainty_factor = torch.sqrt(fused_max_prob * target_max_prob)

        # 综合置信度计算
        confidence = (
                feature_sim_weight * feature_sim +
                pred_sim_weight * pred_sim +
                uncertainty_factor_weight * certainty_factor
        )

        confidence = torch.sigmoid(confidence)

        return confidence

    def get_infodiscri(self,fusion_fea, tar_fea):
        view1_feature = fusion_fea
        view2_feature = tar_fea
        
      
        W = self.W / torch.norm(self.W, p=2, dim=1,
                                keepdim=True)  # Change p to 3, 4, or 5 may result in better performance!
        W = W.to(fusion_fea.device)
        view1_predict = view1_feature.view([view1_feature.shape[0], -1]).mm(W)
        view2_predict = view2_feature.view([view2_feature.shape[0], -1]).mm(W)
                                                          
        view1_Degree = torch.relu(view1_predict)
        view2_Degree = torch.relu(view2_predict)

        view1_cred = self.get_test_category_credibility(view1_Degree)
        view2_cred = self.get_test_category_credibility(view2_Degree)
        view1_uncertainty = self.get_fuzzyUncertainty(view1_cred)
        view2_uncertainty = self.get_fuzzyUncertainty(view2_cred)

        
        ret = dict()
        ret['view1_cred'] = view1_cred
        ret['view2_cred'] = view2_cred
        ret['view1_feature'] = view1_feature
        ret['view2_feature'] = view2_feature
        ret['view1_uncertainty'] = view1_uncertainty
        ret['view2_uncertainty'] = view2_uncertainty
        ret['view1_Degree'] = view1_Degree
        ret['view2_Degree'] = view2_Degree


        return ret

    def get_test_category_credibility(self, Degree):
        top2Degree = torch.topk(Degree, k=2, dim=1, largest=True, sorted=True)[0]
        category_credibility = Degree - top2Degree[:, 0].view([-1, 1]).detach() + 1
        category_credibility += (category_credibility == 1).float() * (top2Degree[:, 0]
                                                                       - top2Degree[:, 1]).reshape(
            [-1, 1]).detach()

        return category_credibility / 2

    def get_fuzzyUncertainty(self, category_credibility):
        nonzero_indices = torch.nonzero(category_credibility)
        class_num = category_credibility.shape[1]
        e = 0.0000001
        if len(nonzero_indices) > 1:
            H = torch.sum((-category_credibility * torch.log(category_credibility + e)
                           - (1 - category_credibility) * torch.log(1 - category_credibility + e)), dim=1, keepdim=True)
            uncertainty = H / (class_num * torch.log(torch.tensor(2)))
        else:
            uncertainty = torch.tensor(0).unsqueeze(0)

        return uncertainty

    def get_discri_loss_retrieval(self,view1_feature, view2_feature,view1_predict,alpha):
        def get_train_category_credibility_for_retrieval(view1_feat, view2_feat):
            view1_norm = torch.nn.functional.normalize(view1_feat, p=2, dim=1)
            view2_norm = torch.nn.functional.normalize(view2_feat, p=2, dim=1)
            view1_norm=view1_norm.unsqueeze(1).unsqueeze(1)
            view2_norm=view2_norm.permute(0, 2, 1)
            similarity_matrix = torch.matmul(view1_norm,view2_norm).squeeze()#.to(query.device)# torch.mm(view1_norm, view2_norm.t())
            similarity_matrix, _ = similarity_matrix.max(-1)
            
            batch_size = similarity_matrix.size(0)
            
            pseudo_labels = torch.eye(batch_size).to(similarity_matrix.device)
            
            predict = torch.softmax(similarity_matrix / 0.07, dim=1)  # 温度缩放
            
            top1Possibility = (predict * (1 - pseudo_labels)).max(1)[0].reshape([-1, 1])
            labelPossibility = (predict * pseudo_labels).max(1)[0].reshape([-1, 1])
            neccessity = (1 - labelPossibility) * (1 - pseudo_labels) + (1 - top1Possibility) * pseudo_labels
            #neccessity = (1 - labelPossibility) * (1 - pseudo_labels)
            r = (predict + neccessity) / 2
            
            return r, pseudo_labels
        
        view1_cred, pseudo_labels1 = get_train_category_credibility_for_retrieval(view1_feature, view2_feature)
        # view2_cred, pseudo_labels2 = get_train_category_credibility_for_retrieval(view2_feature, view1_feature)  # 反向
        eps=1e-7
        view1_cred = torch.clamp(view1_cred, min=eps, max=1-eps)

        labels = torch.arange(view1_cred.shape[0]).long().cuda()
        loss_odl = F.cross_entropy(view1_cred * self.loss_T, labels)
        loss_cl = .0
        return loss_odl, loss_cl

    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)

    
    def robust_infoNCE(self,query, target):
        eps=1e-7
        bs = query.size(0)
        x = torch.matmul(query,target).squeeze().to(query.device)
        sim_i2t,_ = x.max(-1)
        i2t=(sim_i2t/ 0.07).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        
        labels = torch.arange(query.shape[0]).long().cuda()
        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), labels] = 0.   
        loss = - ((1. - i2t).log() * mask).sum() / bs
        return loss
    
    def causal_consistency(self, X, Y):
        def centering(K):
            B, N, _ = K.shape
            unit = torch.ones(N, N, device=K.device) / N
            I = torch.eye(N, device=K.device)
            H = I - unit
            H = H.unsqueeze(0).expand(B, -1, -1)  
            return torch.bmm(torch.bmm(H, K), H)

        # 计算Gram矩阵
        K = torch.bmm(X, X.transpose(-2, -1))  # [B, N, N]
        L = torch.bmm(Y, Y.transpose(-2, -1))  # [B, N, N]

        # 中心化
        K_c = centering(K)
        L_c = centering(L)

        # CKA相似度 
        numerator = (K_c * L_c).sum(dim=(-2, -1))
        denominator = torch.sqrt((K_c * K_c).sum(dim=(-2, -1)) * (L_c * L_c).sum(dim=(-2, -1)))

        cka = numerator / (denominator + 1e-8)
        return 1 - cka.mean()  # 转换为损失
    
    
    def forward(self, samples,device, after_warmup=False):
        if after_warmup:
            image = samples["image"]
            target = samples["target"]
            text = samples["text_input"]

            image_embeds = self.ln_vision(self.visual_encoder(image))

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            # text tokens
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            # fusion reference image and text tokens into a set of multi-modal tokens
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            taregt_embeds = self.ln_vision(self.visual_encoder(target))
            target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=taregt_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True,
            )

            #Target fea
            #[256,32,256]
            target_feats = F.normalize(
                self.vision_proj(target_output.last_hidden_state), dim=-1
            )
            discri_target_feats = torch.mean(target_feats,1)

            #fusion fea
            #[256,256]
            fusion_feats = F.normalize(
                self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
            )
            discri_fusion_feats = fusion_feats
            ret = self.get_infodiscri(discri_fusion_feats,discri_target_feats)

            fusion_feats_=fusion_feats.unsqueeze(1).unsqueeze(1)
            target_feats_=target_feats.permute(0, 2, 1)

            loss_stu_rank=self.robust_infoNCE(fusion_feats_,target_feats_)
            loss_odl,loss_cl = self.get_discri_loss_retrieval(fusion_feats,target_feats,ret['view1_Degree'],0.5)

            return {'loss_stu':loss_stu_rank,
                    'loss_odl':loss_odl,
                   }
        else:
            image = samples["image"]
            target = samples["target"]
            text = samples["text_input"]
            image_intervened = colorful_spectrum_mix(image, target)[0]
            ###============== reference text fusion ===================###
            # reference image feature  
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds_intervened = self.ln_vision(self.visual_encoder(image_intervened))
            #torch.Size([8, 257, 1408])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            image_atts_intervened = torch.ones(image_embeds_intervened.size()[:-1], dtype=torch.long).to(
                image.device
            )   

            # query tokens
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_tokens_intervened = self.query_tokens.expand(image_embeds_intervened.shape[0], -1, -1)

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            query_atts_intervened = torch.ones(query_tokens_intervened.size()[:-1], dtype=torch.long).to(
                image.device
            )   
            # text tokens
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            # fusion reference image and text tokens into a set of multi-modal tokens
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            attention_mask_intervened = torch.cat([query_atts_intervened, text_tokens.attention_mask], dim=1)

            fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            fusion_output_intervened = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens_intervened,
                attention_mask=attention_mask_intervened,
                encoder_hidden_states=image_embeds_intervened,
                encoder_attention_mask=image_atts_intervened,
                return_dict=True,
            )

            taregt_embeds = self.ln_vision(self.visual_encoder(target))
            target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=taregt_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True,
            )

            #Target fea
            #[256,32,256]
            target_feats = F.normalize(
                self.vision_proj(target_output.last_hidden_state), dim=-1
            )
            discri_target_feats = torch.mean(target_feats,1)

            #fusion fea
            #[256,256]
            fusion_feats = F.normalize(
                self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
            )
            fusion_feats_tag = F.normalize(
                self.text_proj_cau(fusion_output.last_hidden_state[:, :32, :]), dim=-1
            )
            fusion_feats_intervened = F.normalize(
                self.text_proj_cau(fusion_output_intervened.last_hidden_state[:, :32, :]), dim=-1
            )   
            # fusion_feats_tag = F.normalize(
            #     self.text_proj_cau(fusion_output.last_hidden_state[:, :32, :]), dim=-1
            # )
            # fusion_feats_intervened = F.normalize(
            #     self.text_proj_cau(fusion_output_intervened.last_hidden_state[:, :32, :]), dim=-1
            # )   

            discri_fusion_feats = fusion_feats
            causal_fusion_feats = fusion_feats_intervened

            ret = self.get_infodiscri(discri_fusion_feats,discri_target_feats)
            # caco = self.causal_consistency(causal_fusion_feats, fusion_feats_tag)

            fusion_feats_=fusion_feats.unsqueeze(1).unsqueeze(1)
            target_feats_=target_feats.permute(0, 2, 1)

            #loss_stu_rank = info_nce(fusion_feats, target_feats)#Eq13
            loss_stu_rank=self.robust_infoNCE(fusion_feats_,target_feats_)
            loss_odl,loss_cl = self.get_discri_loss_retrieval(fusion_feats,target_feats,ret['view1_Degree'],0.5)
            loss_caco = self.causal_consistency(causal_fusion_feats, fusion_feats_tag)
            # loss_caco = F.l1_loss(causal_fusion_feats, fusion_feats_tag)

            return {'loss_stu':loss_stu_rank,
                    'loss_odl':loss_odl,
                    # 'loss_cl':loss_cl,  
                    'loss_caco':loss_caco,
                   }

    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text, return_attns=False):
        reference_embeds = reference_embeds.cuda()
        target_feats = target_feats.cuda()
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp

        if return_attns:
            return sim_i2t, fusion_output.cross_attentions[6].mean(1)

        return sim_i2t
    
    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()

        # return image_embeds
        reference_embeds = image_embeds_frozen

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
        #     attention_mask=attention_mask,
        #     return_dict=True,
        # )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        return fusion_feats.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen


    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
                
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]
        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)



