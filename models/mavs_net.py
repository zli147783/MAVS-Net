"""
MAVS-Net: Modality-Aware Visual-Spatial Fusion Network
for Two-View Correspondence Learning

Core modules:
- VFExtractor: Visual Feature Extractor (ResNet34 + Cross-Attention + MLP)
- SemExtractor: Semantic Extractor (ViT backbone + Cross-Attention)
- OTCFM: Optimal Transport Cross-Modal Fusion (Sinkhorn algorithm)
- SGGA: Semantic-Gated Graph Attention
- SemanticContextTransformer: Two-stage context aggregation with KNN graph + transformer
"""

import math

import timm
import torch
import torch.nn as nn
from models.loss import batch_episym
from models.vanilla_transformer import TransformerLayer
import torch.nn.functional as F
import models.resnet34 as resnet


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        M = X[batch_idx, :, :].squeeze()
        M = M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * 1e-4
        e, v = torch.symeig(M, eigenvectors=True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    """Weighted 8-point algorithm for essential matrix estimation."""
    if logits.shape[1] == 2:
        mask = logits[:, 0, :, 0]
        weights = logits[:, 1, :, 0]

        mask = torch.sigmoid(mask)
        weights = torch.exp(weights) * mask
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    elif logits.shape[1] == 1:
        weights = torch.relu(torch.tanh(logits))  # tanh and relu

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoint eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class ResNetBlock(nn.Module):
    """Residual block with 1x1 convolutions and InstanceNorm + BatchNorm."""

    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNetBlock, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return F.gelu(out)


def knn(x, k):
    """K-Nearest Neighbors based on pairwise distance."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    """Build KNN graph features from point embeddings."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class SGGA(nn.Module):
    """
    Semantic-Gated Graph Attention (SGGA) module.

    Leverages high-level semantic priors to evaluate point-wise semantic
    affinity, actively modulating KNN graph topologies to suppress
    spatial-semantic conflicts. An integrated soft-pooling attention block
    ensures local geometric contexts are captured without discarding
    important structural details.
    """

    def __init__(self, in_channels=128, k=9):
        super(SGGA, self).__init__()
        self.knn_num = k

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
            nn.BatchNorm2d(in_channels),
        )

        self.pointcn = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _semantic_affinity(self, semantic, idx):
        """Compute point-wise semantic affinity for graph gating."""
        B, N, C = semantic.shape
        K = idx.shape[-1]
        semantic_exp = semantic.unsqueeze(2).expand(B, N, K, C)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        neighbor_sem = torch.gather(semantic_exp, dim=1, index=idx_expanded)
        center_sem = semantic.unsqueeze(2)
        affinity = torch.sigmoid((center_sem * neighbor_sem).sum(dim=-1) / math.sqrt(C))
        return affinity

    def forward(self, knn_graph, semantic=None, idx=None):
        knn_graph = self.embed(knn_graph)
        residual = knn_graph

        # Semantic-gated modulation: suppress erroneous topological connections
        if semantic is not None and idx is not None:
            semantic_gate = self._semantic_affinity(semantic, idx)
            knn_graph = knn_graph * semantic_gate.unsqueeze(1)

        knn_graph = self.pointcn(knn_graph)

        # Soft Pooling across the channel dimension
        W = torch.softmax(knn_graph, dim=1)
        F_pool = (knn_graph * W).sum(dim=1, keepdim=True)
        A2 = self.gate_conv(F_pool)
        knn_graph = knn_graph * A2 + residual

        return knn_graph


class AnnularConv(nn.Module):
    """Annular convolution for aggregating KNN graph features."""

    def __init__(self, in_channels=128, k=9):
        super(AnnularConv, self).__init__()
        self.in_channel = in_channels
        self.knn_num = k

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
        if self.knn_num == 6:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )

    def forward(self, features):
        B, C, N, _ = features.shape
        out = self.conv1(features)
        out = self.conv2(out)
        return out


class DiffPool(nn.Module):
    """Differentiable pooling via soft assignment."""

    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class DiffUnpool(nn.Module):
    """Differentiable unpooling via soft assignment."""

    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class SemanticContextTransformer(nn.Module):
    """
    Semantic Context Transformer.

    Integrates SGGA-based local graph attention with global transformer
    context aggregation. Operates in two modes: pruning (predict=False)
    and inlier prediction (predict=True). Supports iterative refinement
    with progressive down-sampling.
    """

    def __init__(self, predict=False, out_channels=128, k_num=9, sampling_rate=0.5,
                 num_heads=4, dropout=None, activation_fn='GELU'):
        super(SemanticContextTransformer, self).__init__()
        self.out_channels = out_channels
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.gab = SGGA(self.out_channels, k=self.k_num)
        self.aggregator = AnnularConv(out_channels, k_num)

        self.encoder = nn.Sequential(
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False)
        )

        self.transformer = TransformerLayer(self.out_channels, num_heads,
                                            dropout=dropout, activation_fn=activation_fn)
        self.resnet_block = ResNetBlock(out_channels * 2, out_channels, pre=True)

        self.linear_0 = nn.Conv2d(out_channels, 1, (1, 1))
        self.linear_1 = nn.Conv2d(out_channels, 1, (1, 1))

        if self.predict == True:
            self.inlier_predictor = nn.Sequential(
                ResNetBlock(self.out_channels, self.out_channels, pre=False),
                nn.Conv2d(self.out_channels, 2, (1, 1))
            )

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, embeddings, x, y, semantic=None):
        B, _, N, _ = x.size()
        src_keypts = x.squeeze(dim=1)[:, :, 0:2]
        tgt_keypts = x.squeeze(dim=1)[:, :, 2:4]
        with torch.no_grad():
            src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
            len_sim = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)

        idx = knn(embeddings.squeeze(-1), k=self.k_num)
        out = get_graph_feature(embeddings, k=self.k_num, idx=idx)
        out = self.gab(out, semantic, idx)
        out = self.aggregator(out)
        out = self.encoder(out)
        w0 = self.linear_0(out).view(B, -1)

        out_g = self.transformer(out.transpose(1, 2).squeeze(dim=-1),
                                  out.transpose(1, 2).squeeze(dim=-1), len_sim)[0]
        out_g = out_g.unsqueeze(dim=-1).transpose(1, 2)
        out = torch.cat([out, out_g], dim=1)
        out = self.resnet_block(out)
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            w2 = self.inlier_predictor(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class CrossAttention(nn.Module):
    """Cross-attention module for fusing features across two views."""

    def __init__(self, channels, head):
        super(CrossAttention, self).__init__()
        self.head = head
        self.head_dim = channels // head

        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)

        self.linear = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_linear = nn.Sequential(
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=1),
            nn.BatchNorm1d(2 * channels), nn.ReLU(),
            nn.Conv1d(2 * channels, channels, kernel_size=1),
        )

    def forward(self, query_tokens, key_tokens, value_tokens=None):
        if value_tokens is None:
            value_tokens = key_tokens
        batch_size = query_tokens.shape[0]

        query = self.q_proj(query_tokens).view(batch_size, self.head, self.head_dim, -1)
        key = self.k_proj(key_tokens).view(batch_size, self.head, self.head_dim, -1)
        value = self.v_proj(value_tokens).view(batch_size, self.head, self.head_dim, -1)

        attention_scores = torch.softmax(
            torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim=-1)
        hidden_states = torch.einsum('bhnm,bhdm->bhdn', attention_scores, value).reshape(
            batch_size, self.head_dim * self.head, -1)
        hidden_states = self.linear(hidden_states)
        output_states = query_tokens + self.cat_linear(
            torch.cat([query_tokens, hidden_states], dim=1))

        return output_states


class SemExtractor(nn.Module):
    """
    Semantic Extractor.

    Extracts high-level semantic priors from two-view images using a
    Vision Transformer (ViT) backbone followed by cross-attention fusion.
    These semantic embeddings guide the SGGA module to resolve
    spatial-semantic conflicts.
    """

    def __init__(self, out_dim, num_heads=4, max_points=2048):
        super(SemExtractor, self).__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                          num_classes=0, global_pool='')
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.cross_attention = CrossAttention(self.backbone.embed_dim, num_heads)
        self.point_attention = CrossAttention(out_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.backbone.embed_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1),
        )
        self.query_embed = nn.Parameter(torch.randn(1, out_dim, max_points))
        self.out_dim = out_dim
        self.max_points = max_points

    def forward(self, img1, img2, num_points):
        if img1.dim() == 5:
            img1 = img1.squeeze(1)
            img2 = img2.squeeze(1)

        # Resize to 224x224 for ViT backbone
        img1 = F.interpolate(img1, size=(224, 224), mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=(224, 224), mode='bilinear', align_corners=False)

        B = img1.shape[0]
        with torch.no_grad():
            feat1 = self.backbone.forward_features(img1)
            feat2 = self.backbone.forward_features(img2)

        tokens1 = feat1[:, 1:, :].permute(0, 2, 1)
        tokens2 = feat2[:, 1:, :].permute(0, 2, 1)
        cross_tokens = self.cross_attention(tokens1, tokens2)
        projected = self.mlp(cross_tokens)

        if num_points > self.max_points:
            raise ValueError(
                f'Requested semantic points {num_points} exceeds max_points {self.max_points}')

        query_tokens = self.query_embed[:, :, :num_points].expand(B, -1, -1)
        semantic_tokens = self.point_attention(query_tokens, projected)
        return semantic_tokens.permute(0, 2, 1)


class OTCFM(nn.Module):
    """
    Optimal Transport Cross-Modal Fusion (OT-CFM).

    Formulates visual-spatial feature interaction as an optimal assignment
    problem solved via the Sinkhorn algorithm, enforcing doubly stochastic
    alignment to suppress ambiguous cross-modal mappings. Modality identity
    biases are injected to alleviate ambiguity during cross-modal interaction.
    """

    def __init__(self, space_dim, hidden_dim, num_heads, dropout=None, activation_fn='GELU'):
        super(OTCFM, self).__init__()
        self.space_dim = space_dim
        self.transformer = TransformerLayer(hidden_dim, num_heads,
                                            dropout=dropout, activation_fn=activation_fn)
        self.diff_pool = DiffPool(hidden_dim, space_dim)
        self.diff_unpool = DiffUnpool(hidden_dim, space_dim)
        self.encoder_spatial = ResNetBlock(hidden_dim, hidden_dim, pre=False)
        self.encoder_visual = ResNetBlock(hidden_dim, hidden_dim, pre=False)
        self.epsilon_v = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.epsilon_s = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.sinkhorn_iters = 10
        self.sinkhorn_epsilon = 0.05

    def log_sinkhorn(self, S):
        """Sinkhorn algorithm for optimal transport alignment."""
        B, M, N = S.shape
        log_mu = -math.log(M)
        log_nu = -math.log(N)
        log_mu = torch.full((B, M), log_mu, device=S.device, dtype=S.dtype)
        log_nu = torch.full((B, N), log_nu, device=S.device, dtype=S.dtype)
        log_K = torch.clamp(S / self.sinkhorn_epsilon, min=-50.0, max=50.0)

        log_u = torch.zeros_like(log_mu)
        log_v = torch.zeros_like(log_nu)

        for _ in range(self.sinkhorn_iters):
            log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            log_v = log_nu - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

        log_pi = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        return torch.exp(log_pi)

    def forward(self, F_v, F_s):
        """
        Args:
            F_v: visual features [B, M, C]
            F_s: spatial features [B, N, C]
        Returns:
            visual_spatial: fused visual-spatial features [B, C, N, 1]
        """
        B, M, C = F_v.shape
        _, N, _ = F_s.shape
        S = torch.matmul(F_v, F_s.transpose(1, 2)) / math.sqrt(C)
        Pi = self.log_sinkhorn(S)
        F_s_prime = torch.matmul(Pi, F_s)

        # Inject learnable modality identity biases
        tilde_F_v = F_v + self.epsilon_v
        tilde_F_s = F_s_prime + self.epsilon_s

        F_f = torch.cat([tilde_F_v, tilde_F_s], dim=1)
        F_f = self.transformer(F_f, F_f, None)[0]
        F_v_prime = F_f[:, :M, :]
        F_s_pp = F_f[:, M:, :]

        initial_sc = F_s.transpose(1, 2).unsqueeze(-1)
        spatial_cues = self.diff_pool(initial_sc)
        spatial_cues = spatial_cues + F_s_pp.transpose(1, 2).unsqueeze(-1)

        visual_cues = (F_v + F_v_prime).transpose(1, 2).unsqueeze(-1)
        spatial_cues = self.encoder_spatial(spatial_cues)
        visual_cues = self.encoder_visual(visual_cues)

        visual_spatial = spatial_cues + visual_cues
        visual_spatial = self.diff_unpool(initial_sc, visual_spatial)
        return visual_spatial


class VFExtractor(nn.Module):
    """
    Visual Feature Extractor (VF-Extractor).

    Extracts visual information from two-view images using a ResNet34
    backbone followed by cross-attention and MLP refinement. The extracted
    visual cues are integrated into correspondences via the OT-CFM module.
    """

    def __init__(self, input_dim, internal_dim, output_dim):
        super(VFExtractor, self).__init__()
        head = 1

        self.cnn = resnet.resnet34(in_channels=3)
        self.cross_attention = CrossAttention(input_dim, head)
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, internal_dim, kernel_size=1),
            nn.BatchNorm1d(internal_dim), nn.ReLU(),
            nn.Conv1d(internal_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim), nn.ReLU(),
        )

    def forward(self, image1, image2):
        img1, img2 = self.cnn(image1.squeeze(dim=1)), self.cnn(image2.squeeze(dim=1))
        B, C, H, W = img1.shape
        img1 = img1.view(B, C, H * W)
        B, C, H, W = img2.shape
        img2 = img2.view(B, C, H * W)
        visual_cues = self.cross_attention(img1, img2)
        visual_cues = self.mlp(visual_cues)

        return visual_cues


class MAVSNet(nn.Module):
    """
    MAVS-Net: Modality-Aware Visual-Spatial Fusion Network.

    A novel framework for two-view correspondence learning that introduces
    optimal transport theory to enforce mathematically grounded cross-modal
    alignment. The network consists of:

    1. SemExtractor: ViT-based semantic feature extraction
    2. VFExtractor: ResNet34-based visual feature extraction
    3. OTCFM: Optimal Transport Cross-Modal Fusion via Sinkhorn algorithm
    4. SemanticContextTransformer (x2): Two-iteration refinement with SGGA

    The first iteration (k=9) prunes correspondences; the second iteration
    (k=6) predicts inlier probabilities and the essential matrix.
    """

    def __init__(self, config):
        super(MAVSNet, self).__init__()
        out_channels = config.net_channels

        self.mlp0 = nn.Sequential(
            nn.Conv2d(4, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(6, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.resnet_blocks0 = nn.Sequential(
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False)
        )
        self.resnet_blocks1 = nn.Sequential(
            ResNetBlock(out_channels * 2, out_channels, pre=True),
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False),
            ResNetBlock(out_channels, out_channels, pre=False)
        )

        self.sem_extractor = SemExtractor(out_channels, num_heads=4, max_points=2048)
        self.vf_extractor = VFExtractor(input_dim=64, internal_dim=96, output_dim=out_channels)
        self.ot_cfm = OTCFM(space_dim=config.space_dim, hidden_dim=out_channels, num_heads=4)
        self.iter0 = SemanticContextTransformer(predict=False, out_channels=out_channels,
                                                 k_num=9, sampling_rate=config.sr)
        self.iter1 = SemanticContextTransformer(predict=True, out_channels=out_channels,
                                                 k_num=6, sampling_rate=config.sr)

    def forward(self, x, y, img1, img2):
        B, _, N, _ = x.shape

        # Stage 1: Extract semantic priors
        semantic = self.sem_extractor(img1, img2, N)

        # Correspondence embedder + first iteration
        iteration_input0 = x.transpose(1, 3).contiguous()
        iteration_input0 = self.mlp0(iteration_input0)
        iteration_input0 = self.resnet_blocks0(iteration_input0)
        x1, y1, ws0, w_ds0 = self.iter0(iteration_input0, x, y, semantic)

        # Stage 2: Visual-spatial fusion with OT-CFM
        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        spatial_cues = self.mlp1(x_.transpose(1, 3).contiguous())
        visual_cues = self.vf_extractor(img1, img2)
        visual_spatial = self.ot_cfm(visual_cues.transpose(1, 2),
                                      spatial_cues.squeeze(-1).transpose(1, 2))
        visual_spatial = torch.cat([spatial_cues, visual_spatial], dim=1)
        iteration_input1 = self.resnet_blocks1(visual_spatial)
        x2, y2, ws1, w_ds1, e_hat = self.iter1(iteration_input1, x_, y1)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat, [x, x, x1, x1, x2]
