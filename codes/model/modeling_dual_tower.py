import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class ImprovedChannelAttention(nn.Module):
    """改进的通道注意力 - 适度平衡约束"""

    def __init__(self, num_channels: int = 2, hidden_dim: int = 512,
                 balance_weight: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.balance_weight = balance_weight

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * num_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_channels),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        self.balance_loss = None

    def forward(self, x):
        b, c, d = x.size()
        x_flat = x.view(b, -1)
        attn_logits = self.attention_net(x_flat)
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)

        if self.training:
            max_weight = attn_weights.max(dim=-1)[0]
            min_weight = attn_weights.min(dim=-1)[0]
            gap = max_weight - min_weight
            self.balance_loss = self.balance_weight * torch.relu(gap - 0.5).mean()
        else:
            self.balance_loss = None

        weighted_x = x * attn_weights.unsqueeze(-1)
        return weighted_x, attn_weights


class MultiModalExpert(nn.Module):
    """单个多模态专家 - 更小的容量强制专业化"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 expert_id: int = 0, dropout: float = 0.15):
        super().__init__()
        self.expert_id = expert_id

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.expert_signature = None

    def forward(self, x):
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        output = self.layer_norm(h)

        if self.training:
            self.expert_signature = output.mean(dim=[0, 1]).detach()

        return output


class SequenceLinearProjector(nn.Module):
    """Non-MoE sequence projector used for w/o MoE ablations."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.15):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.projector(x)


class DenseSequenceProjector(nn.Module):
    """Dense MLP replacement with roughly MoE-matched capacity."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.15):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.projector(x)


class ForcedDiverseMixtureOfExperts(nn.Module):
    """强制多样性的混合专家层 - 使用专家Dropout和硬路由"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_experts: int = 8, top_k: int = 3,
                 load_balance_weight: float = 0.1,
                 diversity_weight: float = 0.1,
                 expert_dropout_rate: float = 0.3,
                 use_expert_dropout: bool = True,
                 use_forced_exploration: bool = True,
                 use_gumbel_noise: bool = True,
                 gumbel_noise_scale: float = 1.0,
                 warmup_steps: int = 2000):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        self.expert_dropout_rate = expert_dropout_rate  # 关键：专家dropout
        self.use_expert_dropout = use_expert_dropout
        self.use_forced_exploration = use_forced_exploration
        self.use_gumbel_noise = use_gumbel_noise
        self.gumbel_noise_scale = gumbel_noise_scale

        self.experts = nn.ModuleList([
            MultiModalExpert(input_dim, hidden_dim, output_dim,
                             expert_id=i, dropout=0.15)
            for i in range(num_experts)
        ])

        self.gate = nn.Linear(input_dim, num_experts)

        nn.init.normal_(self.gate.weight, std=0.001)
        nn.init.zeros_(self.gate.bias)

        self.load_balance_loss = None
        self.diversity_loss = None
        self.latest_expert_fraction = None
        self.latest_expert_importance = None
        self.latest_sample_expert_fraction = None
        self.latest_topk_indices = None

        self.global_step = 0
        self.warmup_steps = warmup_steps  # 前warmup_steps步使用强制探索

    def compute_diversity_loss(self, expert_outputs):
        """计算专家间多样性损失，并保留到专家参数的梯度路径。"""
        if expert_outputs.size(-1) < 2:
            return torch.tensor(0.0, device=expert_outputs.device)

        signatures = expert_outputs.mean(dim=(0, 1)).transpose(0, 1)
        signatures_norm = F.normalize(signatures, p=2, dim=1)
        similarity_matrix = torch.matmul(signatures_norm, signatures_norm.t())

        mask = torch.eye(len(signatures), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)

        diversity_loss = similarity_matrix.abs().mean()

        return diversity_loss

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        available_experts = list(range(self.num_experts))
        if self.training and self.use_expert_dropout and self.global_step > self.warmup_steps:
            num_dropout = min(
                int(self.num_experts * self.expert_dropout_rate),
                max(0, self.num_experts - self.top_k)
            )
            dropout_experts = torch.randperm(self.num_experts)[:num_dropout].tolist()
            available_experts = [e for e in range(self.num_experts) if e not in dropout_experts]

        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]

        if self.training and self.use_expert_dropout and self.global_step > self.warmup_steps:
            for expert_id in range(self.num_experts):
                if expert_id not in available_experts:
                    gate_logits[:, :, expert_id] = -1e9

        if self.training and self.use_forced_exploration and self.global_step <= self.warmup_steps:
            topk_indices = torch.zeros(batch_size, seq_len, self.top_k,
                                       dtype=torch.long, device=x.device)
            for b in range(batch_size):
                for s in range(seq_len):
                    perm = torch.randperm(self.num_experts, device=x.device)[:self.top_k]
                    topk_indices[b, s] = perm

            topk_scores = torch.ones(batch_size, seq_len, self.top_k, device=x.device) / self.top_k
            gate_scores = F.softmax(gate_logits, dim=-1)

        else:
            if self.training and self.use_gumbel_noise:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(gate_logits) + 1e-8) + 1e-8)
                gate_logits = gate_logits + gumbel_noise * self.gumbel_noise_scale

            gate_scores = F.softmax(gate_logits, dim=-1)

            topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        expert_mask = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        expert_mask.scatter_(2, topk_indices, 1.0)
        expert_fraction = expert_mask.mean(dim=[0, 1])
        expert_importance = gate_scores.mean(dim=[0, 1])
        self.latest_expert_fraction = expert_fraction.detach()
        self.latest_expert_importance = expert_importance.detach()
        self.latest_sample_expert_fraction = expert_mask.mean(dim=1).detach()
        self.latest_topk_indices = topk_indices.detach()

        if self.training:
            target_usage = 1.0 / self.num_experts

            switch_loss = self.num_experts * (expert_fraction * expert_importance).sum()

            variance_loss = ((expert_fraction - target_usage) ** 2).sum() * 10.0

            min_threshold = target_usage * 0.5
            underused_penalty = torch.relu(min_threshold - expert_fraction).sum() * 20.0

            max_threshold = target_usage * 2.0
            overused_penalty = torch.relu(expert_fraction - max_threshold).sum() * 20.0

            entropy = -(gate_scores * torch.log(gate_scores + 1e-8)).sum(dim=-1).mean()
            max_entropy = math.log(self.num_experts)
            entropy_loss = -0.2 * (entropy / max_entropy)

            self.load_balance_loss = self.load_balance_weight * (
                    switch_loss +
                    variance_loss +
                    underused_penalty +
                    overused_penalty +
                    entropy_loss
            )
        else:
            self.load_balance_loss = None

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)

        if self.training:
            self.diversity_loss = self.diversity_weight * self.compute_diversity_loss(expert_outputs)
        else:
            self.diversity_loss = None

        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, seq_len, self.top_k)
        seq_indices = torch.arange(seq_len, device=x.device).view(1, -1, 1).expand(batch_size, -1, self.top_k)
        selected_outputs = expert_outputs[batch_indices, seq_indices, :, topk_indices]
        weighted_output = (selected_outputs * topk_scores.unsqueeze(-1)).sum(dim=2)

        if self.training:
            self.global_step += 1

        return weighted_output


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.out_proj(context)
        output = self.layer_norm(output + query)
        return output


class MultiModalFusionTower(nn.Module):
    """多模态融合塔"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.text_proj = nn.Linear(args.model.text_encoder_dim, args.model.unified_dim)
        self.image_proj = nn.Linear(args.model.image_encoder_dim, args.model.unified_dim)
        self.emb_image_proj = nn.Linear(args.model.emd_image_encoder_dim, args.model.unified_dim)

        load_balance_weight = args.model.get('load_balance_weight', 0.1)
        diversity_weight = args.model.get('diversity_weight', 0.1)
        expert_dropout_rate = args.model.get('expert_dropout_rate', 0.3)
        self.expert_layer_type = args.model.get('expert_layer_type', 'moe')
        self.use_cross_attention = args.model.get('use_cross_attention', True)

        if self.expert_layer_type == 'moe':
            moe_kwargs = dict(
                input_dim=args.model.unified_dim,
                hidden_dim=args.model.expert_hidden_dim,
                output_dim=args.model.unified_dim,
                num_experts=args.model.num_experts,
                top_k=args.model.top_k_experts,
                load_balance_weight=load_balance_weight,
                diversity_weight=diversity_weight,
                expert_dropout_rate=expert_dropout_rate,
                use_expert_dropout=args.model.get('use_expert_dropout', True),
                use_forced_exploration=args.model.get('use_forced_exploration', True),
                use_gumbel_noise=args.model.get('use_gumbel_noise', True),
                gumbel_noise_scale=args.model.get('gumbel_noise_scale', 1.0),
                warmup_steps=args.model.get('moe_warmup_steps', 2000),
            )
            self.text_moe = ForcedDiverseMixtureOfExperts(**moe_kwargs)
            self.image_moe = ForcedDiverseMixtureOfExperts(**moe_kwargs)
        elif self.expert_layer_type == 'dense':
            dense_hidden_dim = args.model.get(
                'dense_hidden_dim',
                args.model.expert_hidden_dim * args.model.num_experts
            )
            self.text_moe = DenseSequenceProjector(
                args.model.unified_dim,
                dense_hidden_dim,
                args.model.unified_dim,
                dropout=args.model.get('dense_dropout', 0.15)
            )
            self.image_moe = DenseSequenceProjector(
                args.model.unified_dim,
                dense_hidden_dim,
                args.model.unified_dim,
                dropout=args.model.get('dense_dropout', 0.15)
            )
        elif self.expert_layer_type == 'linear':
            self.text_moe = SequenceLinearProjector(
                args.model.unified_dim,
                args.model.unified_dim,
                dropout=args.model.get('dense_dropout', 0.15)
            )
            self.image_moe = SequenceLinearProjector(
                args.model.unified_dim,
                args.model.unified_dim,
                dropout=args.model.get('dense_dropout', 0.15)
            )
        else:
            raise ValueError(f"Unsupported expert_layer_type: {self.expert_layer_type}")

        self.text_to_image_attn = CrossModalAttention(
            args.model.unified_dim,
            args.model.num_attention_heads,
            args.model.attention_dropout
        )

        self.image_to_text_attn = CrossModalAttention(
            args.model.unified_dim,
            args.model.num_attention_heads,
            args.model.attention_dropout
        )

        if args.model.use_senet:
            channel_balance_weight = args.model.get('channel_balance_weight', 0.1)
            self.channel_attention = ImprovedChannelAttention(
                num_channels=2,
                hidden_dim=args.model.unified_dim,
                balance_weight=channel_balance_weight
            )

        self.fusion_fc = nn.Sequential(
            nn.Linear(args.model.unified_dim * 2, args.model.unified_dim),
            nn.GELU(),
            nn.Dropout(args.model.fusion_dropout),
            nn.Linear(args.model.unified_dim, args.model.unified_dim)
        )

        self.final_norm = nn.LayerNorm(args.model.unified_dim)

    def forward(self, text_embeds, image_embeds, text_tokens=None, image_patches=None,
                apply_modal_dropout=False):
        text_feat = self.text_proj(text_embeds)
        image_feat = self.emb_image_proj(image_embeds)

        if apply_modal_dropout and self.training:
            drop_prob = self.args.model.modal_dropout_prob
            text_mask = torch.bernoulli(torch.full_like(text_feat[:, :1], 1 - drop_prob))
            image_mask = torch.bernoulli(torch.full_like(image_feat[:, :1], 1 - drop_prob))
            text_feat = text_feat * text_mask
            image_feat = image_feat * image_mask

        if text_tokens is not None and image_patches is not None:
            text_seq = self.text_proj(text_tokens)
            image_seq = self.image_proj(image_patches)

            text_seq = self.text_moe(text_seq)
            image_seq = self.image_moe(image_seq)

            if self.use_cross_attention:
                text_enhanced = self.text_to_image_attn(text_seq, image_seq, image_seq)
                image_enhanced = self.image_to_text_attn(image_seq, text_seq, text_seq)
            else:
                text_enhanced = text_seq
                image_enhanced = image_seq

            text_feat = text_enhanced.mean(dim=1)
            image_feat = image_enhanced.mean(dim=1)

        channel_attn_weights = None
        if self.args.model.use_senet:
            stacked = torch.stack([text_feat, image_feat], dim=1)
            stacked, channel_attn_weights = self.channel_attention(stacked)
            text_feat, image_feat = stacked[:, 0], stacked[:, 1]

        fused = torch.cat([text_feat, image_feat], dim=-1)
        output = self.fusion_fc(fused)
        output = self.final_norm(output)
        output = F.normalize(output, p=2, dim=-1)

        return output, channel_attn_weights


class DualTowerEncoder(nn.Module):
    """双塔编码器"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        print("===加载CLIP===")
        print(args.pretrained_model)

        self.clip = CLIPModel.from_pretrained(args.pretrained_model)
        self.mention_tower = MultiModalFusionTower(args)

        if args.model.share_tower_params:
            self.entity_tower = self.mention_tower
        else:
            self.entity_tower = MultiModalFusionTower(args)

        self.text_dropout = nn.Dropout(args.model.text_dropout)
        self.image_dropout = nn.Dropout(args.model.image_dropout)

    def encode_mention(self, input_ids, attention_mask, pixel_values,
                       return_sequences=True, apply_modal_dropout=False):
        clip_output = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        text_embeds = self.text_dropout(clip_output.text_embeds)
        image_embeds = self.image_dropout(clip_output.image_embeds)

        text_tokens = None
        image_patches = None
        if return_sequences:
            text_tokens = self.text_dropout(clip_output.text_model_output[0])
            image_patches = self.image_dropout(clip_output.vision_model_output[0])

        mention_embedding, _ = self.mention_tower(
            text_embeds, image_embeds, text_tokens, image_patches, apply_modal_dropout
        )

        return mention_embedding

    def encode_entity(self, input_ids, attention_mask, pixel_values,
                      return_sequences=True, apply_modal_dropout=False):
        clip_output = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        text_embeds = self.text_dropout(clip_output.text_embeds)
        image_embeds = self.image_dropout(clip_output.image_embeds)

        text_tokens = None
        image_patches = None
        if return_sequences:
            text_tokens = self.text_dropout(clip_output.text_model_output[0])
            image_patches = self.image_dropout(clip_output.vision_model_output[0])

        entity_embedding, _ = self.entity_tower(
            text_embeds, image_embeds, text_tokens, image_patches, apply_modal_dropout
        )

        return entity_embedding

    def forward(self, mention_batch, entity_batch, apply_modal_dropout=False):
        clip_output_mention = self.clip(
            input_ids=mention_batch['input_ids'],
            attention_mask=mention_batch['attention_mask'],
            pixel_values=mention_batch['pixel_values']
        )

        text_embeds_m = self.text_dropout(clip_output_mention.text_embeds)
        image_embeds_m = self.image_dropout(clip_output_mention.image_embeds)
        text_tokens_m = self.text_dropout(clip_output_mention.text_model_output[0])
        image_patches_m = self.image_dropout(clip_output_mention.vision_model_output[0])

        mention_embeddings, mention_channel_weights = self.mention_tower(
            text_embeds_m, image_embeds_m, text_tokens_m, image_patches_m, apply_modal_dropout
        )

        clip_output_entity = self.clip(
            input_ids=entity_batch['input_ids'],
            attention_mask=entity_batch['attention_mask'],
            pixel_values=entity_batch['pixel_values']
        )

        text_embeds_e = self.text_dropout(clip_output_entity.text_embeds)
        image_embeds_e = self.image_dropout(clip_output_entity.image_embeds)
        text_tokens_e = self.text_dropout(clip_output_entity.text_model_output[0])
        image_patches_e = self.image_dropout(clip_output_entity.vision_model_output[0])

        entity_embeddings, entity_channel_weights = self.entity_tower(
            text_embeds_e, image_embeds_e, text_tokens_e, image_patches_e, apply_modal_dropout
        )

        return mention_embeddings, entity_embeddings, mention_channel_weights, entity_channel_weights


class DualTowerMatcher(nn.Module):
    """双塔匹配器"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = args.model.temperature

    def forward(self, mention_embeddings, entity_embeddings):
        similarity = torch.matmul(mention_embeddings, entity_embeddings.t())
        similarity = similarity / self.temperature
        return similarity
