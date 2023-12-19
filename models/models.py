import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, dur, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, length, d_tensor = k.size() # head

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(1, 2)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product # [64, 900, 900]
        
        _, score_size, _ = score.size()
        
        mask = torch.ones(batch_size, score_size, score_size).cuda()
        
        for i in range(batch_size):
            # mask[i] = torch.repeat(1, dur[i], dur[i]).cuda()
            temp = torch.ones(dur[i], dur[i]).cuda()
            mask[i] = F.pad(temp, (0, 900 - dur[i], 0, 900 - dur[i]), "constant", 0)
        
        # mask = torch.ones(batch_size, dur)

        score = torch.mul(score, mask)

        # 2. apply masking (opt)
        # if mask is not None:
        score = score.masked_fill(mask == 0, -e) # -10000

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
    

class CLNet(nn.Module):
    def __init__(self, input_dim, output_dim, dynamic_pooling=True):
        super(CLNet, self).__init__()
        self.attention = ScaleDotProductAttention()
        self.d_model = 40
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        
        doppler_pooling = np.zeros((4,), dtype=np.int32)
        time_pooling = np.zeros((4,), dtype=np.int32)
        self.input_dim = input_dim
        # dynmiac poolling considers the features informations in the receptive fields
        # The features are obtained from doppler signals
        if dynamic_pooling:
            doppler_pooling[:] = int((float(input_dim[2]) / 5) ** .25)
            res = ((float(input_dim[2]) / 5) ** .25) - doppler_pooling[0]
            for i in range(int(round(res * 4))):
                doppler_pooling[i] += 1
        else:
            doppler_pooling[:] = 2;
            c = 0
            while input_dim[2] < np.prod(doppler_pooling):
                doppler_pooling[-(c % 4) - 1] -= 1;
                c += 1

        if dynamic_pooling:
            time_pooling[:] = int((float(input_dim[1]) / 5) ** .25)
            res = ((float(input_dim[1]) / 5) ** .25) - time_pooling[0]
            for i in range(int(round(res * 4))):
                time_pooling[i] += 1
        else:
            time_pooling[:] = 2;
            c = 0
            while input_dim[1] < np.prod(time_pooling):
                time_pooling[-(c % 4) - 1] -= 1;
                c += 1

        self.encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=input_dim[0], out_channels=4, kernel_size=3, padding=1), # input_dim = input_dim=(1, 45, 205)
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[0], doppler_pooling[0])),
            # Conv2
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[1], doppler_pooling[1])),
            # Conv3
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[2], doppler_pooling[2])),
            # Conv4
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[3], doppler_pooling[3]))
        )  # --- Conv sequential ends ---
        
        self.classifier_1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 40)
        )

        self.embedder_1 = nn.Sequential(
            nn.Linear(45,40),
            nn.ELU(inplace=True)
        )
        
        self.st_classifier = nn.Sequential(
            nn.Linear(40, 10),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(10, 1)
        )
        
        self.ed_classifier = nn.Sequential(
            nn.Linear(40, 10),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(10, 1)
        ) 

        self.frame_level_classifier = nn.Sequential(
            nn.Linear(40, 10),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(10, 1)   
        )

    def spatial_encoder(self, x, U): # x.shape = [64, 1, 900, 205]
        B, _, L, D = x.shape # (64, 1, 900, 205)
        w = int(U/2) # w = 22, U = 45
        z = torch.zeros((B,1,22,D), device=x.device) # torch.Size([64, 1, 22, 205])
        x_ = torch.zeros((B,L,16*5*5), device=x.device) # torch.Size([64, 900, 400]) #ANCHOR: 16*5*5
        
        for i in range(L):
            if i < w:
                y = torch.cat([z[:,:,:U-(i+w+1),:], x[:, :, :i+w+1, :]], dim=2) # [64, 1, 21, 205], [64, 1, 24, 205], y = [64, 1, 45, 205]
                x_[:,i,:] = self.encoder(y).view(B,-1) # [64, 400]
            elif w <= i and i < L - w:
                y = x[:, :, i-w:i+w+1, :]
                x_[:,i,:] = self.encoder(y).view(B,-1)
            else:
                y = torch.cat([z[:,:,:U-(L-i+w),:], x[:, :, i-w:, :]], dim=2)
                x_[:,i,:] = self.encoder(y).view(B,-1)

        return x_ # [64, 900, 400]

    def causal_attention(self, x, dur):
        q, k, v = x, x, x
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, dur)
        return out, attention

    def forward(self, z, targets_onehot, dur):
        B, _, L, D = z.shape # torch.Size([64, 1, 900, 205])
        U = self.input_dim[1] # (1, 45, 205) -> U = 45

        # z_p: Patch-wise Doppler Encoding
        z_p_a = self.spatial_encoder(z, U) # [64, 900, 400]
        z_p = self.classifier_1(z_p_a) # [64, 900, 40]

        # t: One hot encoding of input target        
        t = targets_onehot.unsqueeze(dim=1).repeat(1, 900, 1).to("cuda") # 64, 1, 5

        # z_t: Target-guided Doppler Encoding
        z_t_a = torch.concat((z_p, t), dim=2) # [64, 900, 405] => [64, 900, 45]
        z_t = self.embedder_1(z_t_a)
        
        # x: Final Localization Doppler Feature
        x_a, _ = self.causal_attention(z_t, dur) # [64, 900, 405] => [64, 900, 45]
        x = x_a + z_t

        # Causal Localization Head
        fl = self.frame_level_classifier(x).squeeze(dim=2) # torch.flatten(x_a, start_dim=1) # [64, 900, 45] => [64, 900, 1]

        st = self.st_classifier(x).squeeze(dim=2)

        ed = self.ed_classifier(x).squeeze(dim=2)

        return fl, st, ed # [64, 900, 1] here







