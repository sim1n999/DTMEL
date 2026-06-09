import math
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from codes.model.modeling_dual_tower import DualTowerEncoder, DualTowerMatcher


class LightningDualTower(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.encoder = DualTowerEncoder(args)
        self.matcher = DualTowerMatcher(args)

        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.use_memory_bank = bool(self.args.model.get('use_memory_bank', False))
        self.memory_bank_size = int(self.args.model.get('memory_bank_size', 0))
        self.memory_bank_warmup_steps = int(self.args.model.get('memory_bank_warmup_steps', 0))
        bank_dim = int(self.args.model.get('unified_dim', 512))
        self.register_buffer('entity_memory_bank', torch.empty(0, bank_dim), persistent=False)

        self.entity_embeddings = None

        self.channel_weights_history = []

    def get_entity_memory_bank_negatives(self):
        if not self.use_memory_bank or self.memory_bank_size <= 0:
            return None
        if self.global_step < self.memory_bank_warmup_steps:
            return None
        if self.entity_memory_bank.numel() == 0:
            return None
        return self.entity_memory_bank.detach()

    @torch.no_grad()
    def enqueue_entity_memory_bank(self, entity_embeddings):
        if not self.use_memory_bank or self.memory_bank_size <= 0:
            return

        embeddings = entity_embeddings.detach()
        if embeddings.dim() != 2:
            embeddings = embeddings.reshape(embeddings.size(0), -1)

        if self.entity_memory_bank.numel() > 0 and self.entity_memory_bank.size(1) != embeddings.size(1):
            self.entity_memory_bank = torch.empty(0, embeddings.size(1), device=embeddings.device)

        if embeddings.size(0) >= self.memory_bank_size:
            self.entity_memory_bank = embeddings[-self.memory_bank_size:].detach()
            return

        if self.entity_memory_bank.numel() == 0:
            updated_bank = embeddings
        else:
            updated_bank = torch.cat([self.entity_memory_bank.to(embeddings.device), embeddings], dim=0)

        self.entity_memory_bank = updated_bank[-self.memory_bank_size:].detach()

    def info_nce_loss(self, mention_embeddings, entity_embeddings, temperature=0.07,
                      extra_entity_negatives=None):
        """
        InfoNCE损失（对比学习）
        Args:
            mention_embeddings: [batch_size, dim]
            entity_embeddings: [batch_size, dim]
            extra_entity_negatives: optional stale entity embeddings from previous batches
        """
        batch_size = mention_embeddings.size(0)

        in_batch_logits = torch.matmul(mention_embeddings, entity_embeddings.t()) / temperature
        logits = in_batch_logits
        if extra_entity_negatives is not None and extra_entity_negatives.numel() > 0:
            extra_entity_negatives = extra_entity_negatives.to(mention_embeddings.device)
            memory_logits = torch.matmul(mention_embeddings, extra_entity_negatives.t()) / temperature
            logits = torch.cat([in_batch_logits, memory_logits], dim=1)

        labels = torch.arange(batch_size, device=logits.device)

        loss_m2e = self.loss_fct(logits, labels)  # mention -> entity

        if self.args.model.bidirectional_loss:
            loss_e2m = self.loss_fct(in_batch_logits.t(), labels)  # entity -> mention
            loss = (self.args.model.mention_to_entity_weight * loss_m2e +
                    self.args.model.entity_to_mention_weight * loss_e2m)
        else:
            loss = loss_m2e

        return loss, logits


    def collect_load_balance_loss(self):
        """收集所有MoE模块的负载均衡损失"""
        lb_loss = 0.0

        if hasattr(self.encoder.mention_tower.text_moe, 'load_balance_loss'):
            if self.encoder.mention_tower.text_moe.load_balance_loss is not None:
                lb_loss += self.encoder.mention_tower.text_moe.load_balance_loss

        if hasattr(self.encoder.mention_tower.image_moe, 'load_balance_loss'):
            if self.encoder.mention_tower.image_moe.load_balance_loss is not None:
                lb_loss += self.encoder.mention_tower.image_moe.load_balance_loss

        if not self.args.model.share_tower_params:
            if hasattr(self.encoder.entity_tower.text_moe, 'load_balance_loss'):
                if self.encoder.entity_tower.text_moe.load_balance_loss is not None:
                    lb_loss += self.encoder.entity_tower.text_moe.load_balance_loss

            if hasattr(self.encoder.entity_tower.image_moe, 'load_balance_loss'):
                if self.encoder.entity_tower.image_moe.load_balance_loss is not None:
                    lb_loss += self.encoder.entity_tower.image_moe.load_balance_loss

        return lb_loss

    def collect_diversity_loss(self):
        """收集所有MoE模块的专家多样性损失（新增）"""
        div_loss = 0.0

        if hasattr(self.encoder.mention_tower.text_moe, 'diversity_loss'):
            if self.encoder.mention_tower.text_moe.diversity_loss is not None:
                div_loss += self.encoder.mention_tower.text_moe.diversity_loss

        if hasattr(self.encoder.mention_tower.image_moe, 'diversity_loss'):
            if self.encoder.mention_tower.image_moe.diversity_loss is not None:
                div_loss += self.encoder.mention_tower.image_moe.diversity_loss

        if not self.args.model.share_tower_params:
            if hasattr(self.encoder.entity_tower.text_moe, 'diversity_loss'):
                if self.encoder.entity_tower.text_moe.diversity_loss is not None:
                    div_loss += self.encoder.entity_tower.text_moe.diversity_loss

            if hasattr(self.encoder.entity_tower.image_moe, 'diversity_loss'):
                if self.encoder.entity_tower.image_moe.diversity_loss is not None:
                    div_loss += self.encoder.entity_tower.image_moe.diversity_loss

        return div_loss

    def collect_channel_balance_loss(self):
        """收集通道注意力的平衡损失"""
        ch_loss = 0.0

        if hasattr(self.encoder.mention_tower, 'channel_attention'):
            if hasattr(self.encoder.mention_tower.channel_attention, 'balance_loss'):
                if self.encoder.mention_tower.channel_attention.balance_loss is not None:
                    ch_loss += self.encoder.mention_tower.channel_attention.balance_loss

        if not self.args.model.share_tower_params:
            if hasattr(self.encoder.entity_tower, 'channel_attention'):
                if hasattr(self.encoder.entity_tower.channel_attention, 'balance_loss'):
                    if self.encoder.entity_tower.channel_attention.balance_loss is not None:
                        ch_loss += self.encoder.entity_tower.channel_attention.balance_loss

        return ch_loss

    def log_expert_usage(self, batch_idx):
        """记录MoE路由中间统计，供collapse分析和机制消融使用。"""
        log_every = self.args.model.get('expert_log_every_n_steps', 100)
        if log_every <= 0 or batch_idx % log_every != 0:
            return

        towers = [('shared', self.encoder.mention_tower)]
        if not self.args.model.share_tower_params:
            towers = [
                ('mention', self.encoder.mention_tower),
                ('entity', self.encoder.entity_tower),
            ]

        for tower_name, tower in towers:
            for modality in ('text', 'image'):
                module = getattr(tower, f'{modality}_moe', None)
                expert_fraction = getattr(module, 'latest_expert_fraction', None)
                expert_importance = getattr(module, 'latest_expert_importance', None)

                if expert_fraction is not None:
                    for idx, value in enumerate(expert_fraction):
                        self.log(
                            f'Train/{tower_name}_{modality}_expert{idx}_fraction',
                            value,
                            on_step=True,
                            on_epoch=True,
                        )

                if expert_importance is not None:
                    for idx, value in enumerate(expert_importance):
                        self.log(
                            f'Train/{tower_name}_{modality}_expert{idx}_importance',
                            value,
                            on_step=True,
                            on_epoch=True,
                        )

    def training_step(self, batch, batch_idx):
        mention_batch = {}
        entity_batch = {}
        for k, v in batch.items():
            if k.startswith('ent_'):
                entity_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v

        _ = entity_batch.pop('empty_img_flag', None)

        mention_embeddings, entity_embeddings, mention_ch_weights, entity_ch_weights = self.encoder(
            mention_batch, entity_batch, apply_modal_dropout=True
        )

        extra_entity_negatives = self.get_entity_memory_bank_negatives()
        loss, logits = self.info_nce_loss(
            mention_embeddings,
            entity_embeddings,
            self.args.model.temperature,
            extra_entity_negatives=extra_entity_negatives
        )

        lb_loss = self.collect_load_balance_loss()

        ch_loss = self.collect_channel_balance_loss()

        div_loss = self.collect_diversity_loss()

        total_loss = loss + lb_loss + div_loss + ch_loss

        preds = torch.argmax(logits, dim=1)
        labels = torch.arange(mention_embeddings.size(0), device=logits.device)
        acc = (preds == labels).float().mean()

        self.log('Train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('Train/lb_loss', lb_loss.detach() if isinstance(lb_loss, torch.Tensor) else 0.0,
                 on_step=True, on_epoch=True)
        self.log('Train/div_loss', div_loss.detach() if isinstance(div_loss, torch.Tensor) else 0.0,
                 on_step=True, on_epoch=True)  # 新增
        self.log('Train/ch_loss', ch_loss.detach() if isinstance(ch_loss, torch.Tensor) else 0.0,
                 on_step=True, on_epoch=True)
        self.log('Train/total_loss', total_loss.detach(), on_step=True, on_epoch=True)
        self.log('Train/acc', acc.detach(), on_step=True, on_epoch=True)
        if self.use_memory_bank:
            self.log('Train/memory_bank_size', float(self.entity_memory_bank.size(0)),
                     on_step=True, on_epoch=True)
            self.log('Train/negatives_per_query', float(logits.size(1) - 1),
                     on_step=True, on_epoch=True)

        if batch_idx % 100 == 0 and mention_ch_weights is not None:
            avg_mention_weights = mention_ch_weights.mean(dim=0)  # [2]
            avg_entity_weights = entity_ch_weights.mean(dim=0) if entity_ch_weights is not None else None

            self.log('Train/mention_text_weight', avg_mention_weights[0].detach(), on_step=True)
            self.log('Train/mention_image_weight', avg_mention_weights[1].detach(), on_step=True)

            if avg_entity_weights is not None:
                self.log('Train/entity_text_weight', avg_entity_weights[0].detach(), on_step=True)
                self.log('Train/entity_image_weight', avg_entity_weights[1].detach(), on_step=True)

        self.log_expert_usage(batch_idx)
        self.enqueue_entity_memory_bank(entity_embeddings)

        return total_loss

    def validation_step(self, batch, batch_idx):
        answer = batch.pop('answer')
        batch_size = len(answer)

        mention_embeddings = self.encoder.encode_mention(
            batch['input_ids'],
            batch['attention_mask'],
            batch['pixel_values'],
            return_sequences=True,
            apply_modal_dropout=False
        )

        scores = []
        chunk_size = self.args.data.eval_chunk_size
        num_chunks = math.ceil(self.args.data.num_entity / chunk_size)

        for idx in range(num_chunks):
            start_pos = idx * chunk_size
            end_pos = min((idx + 1) * chunk_size, self.args.data.num_entity)

            chunk_entity_embeddings = self.entity_embeddings[start_pos:end_pos].to(mention_embeddings.device)

            chunk_scores = self.matcher(mention_embeddings, chunk_entity_embeddings)
            scores.append(chunk_scores)

        scores = torch.cat(scores, dim=-1)  # [batch_size, num_entities]

        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()

        return {'rank': tgt_rank, 'scores': scores.detach().cpu().numpy()}

    def on_validation_start(self):
        """验证开始前：编码所有实体"""
        print("\n[验证开始] 正在编码所有实体...")
        entity_dataloader = self.trainer.datamodule.entity_dataloader()

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='编码实体', total=len(entity_dataloader)):
                batch = pl.utilities.move_data_to_device(batch, self.device)

                entity_embeddings = self.encoder.encode_entity(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['pixel_values'],
                    return_sequences=True,
                    apply_modal_dropout=False
                )

                all_embeddings.append(entity_embeddings.cpu())

        self.entity_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"[验证开始] 实体编码完成,总数: {self.entity_embeddings.size(0)}\n")

    def validation_epoch_end(self, outputs):
        """验证结束：计算指标"""
        self.entity_embeddings = None

        ranks = np.concatenate([out['rank'].numpy() for out in outputs])

        hits1 = (ranks <= 1).mean()
        hits3 = (ranks <= 3).mean()
        hits5 = (ranks <= 5).mean()
        hits10 = (ranks <= 10).mean()
        hits20 = (ranks <= 20).mean()
        hits50 = (ranks <= 50).mean()
        mr = ranks.mean()
        mrr = (1.0 / ranks).mean()

        self.log("Val/hits@1", hits1)
        self.log("Val/hits@3", hits3)
        self.log("Val/hits@5", hits5)
        self.log("Val/hits@10", hits10)
        self.log("Val/hits@20", hits20)
        self.log("Val/hits@50", hits50)
        self.log("Val/mr", mr)
        self.log("Val/mrr", mrr)

        print(f"\n{'=' * 60}")
        print(f"验证结果:")
        print(f"  MRR: {mrr:.4f} | MR: {mr:.2f}")
        print(f"  Hits@1: {hits1:.4f} | Hits@3: {hits3:.4f} | Hits@5: {hits5:.4f}")
        print(f"  Hits@10: {hits10:.4f} | Hits@20: {hits20:.4f} | Hits@50: {hits50:.4f}")
        print(f"{'=' * 60}\n")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_start(self):
        """测试开始前：编码所有实体"""
        print("\n[测试开始] 正在编码所有实体...")
        entity_dataloader = self.trainer.datamodule.entity_dataloader()

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='编码实体', total=len(entity_dataloader)):
                batch = pl.utilities.move_data_to_device(batch, self.device)

                entity_embeddings = self.encoder.encode_entity(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['pixel_values'],
                    return_sequences=True,
                    apply_modal_dropout=False
                )

                all_embeddings.append(entity_embeddings.cpu())

        self.entity_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"[测试开始] 实体编码完成,总数: {self.entity_embeddings.size(0)}\n")

    def test_epoch_end(self, outputs):
        """测试结束：计算指标"""
        self.entity_embeddings = None

        ranks = np.concatenate([out['rank'].numpy() for out in outputs])

        hits1 = (ranks <= 1).mean()
        hits3 = (ranks <= 3).mean()
        hits5 = (ranks <= 5).mean()
        hits10 = (ranks <= 10).mean()
        hits20 = (ranks <= 20).mean()
        hits50 = (ranks <= 50).mean()
        mr = ranks.mean()
        mrr = (1.0 / ranks).mean()

        self.log("Test/hits@1", hits1)
        self.log("Test/hits@3", hits3)
        self.log("Test/hits@5", hits5)
        self.log("Test/hits@10", hits10)
        self.log("Test/hits@20", hits20)
        self.log("Test/hits@50", hits50)
        self.log("Test/mr", mr)
        self.log("Test/mrr", mrr)

        print(f"\n{'=' * 60}")
        print(f"测试结果:")
        print(f"  MRR: {mrr:.4f} | MR: {mr:.2f}")
        print(f"  Hits@1: {hits1:.4f} | Hits@3: {hits3:.4f} | Hits@5: {hits5:.4f}")
        print(f"  Hits@10: {hits10:.4f} | Hits@20: {hits20:.4f} | Hits@50: {hits50:.4f}")
        print(f"{'=' * 60}\n")

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.args.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
