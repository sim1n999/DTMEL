import os
import copy
import json
import random
import pickle
import torch
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from urllib.parse import unquote

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_json_file(filepath):
    data = []
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
            data.extend(d)
    elif isinstance(filepath, list):
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                data.extend(d)
    return data


class DataModuleForDualTower(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.base_path = self.args.data.get('base_path', '')
        self.tokenizer = CLIPProcessor.from_pretrained(
            self._resolve_path(self.args.pretrained_model)
        ).tokenizer
        self.image_processor = CLIPProcessor.from_pretrained(
            self._resolve_path(self.args.pretrained_model)
        ).feature_extractor

        with open(self._resolve_path(self.args.data.qid2id), 'r', encoding='utf-8') as f:
            self.qid2id = json.loads(f.readline())

        self.raw_kb_entity = sorted(
            _load_json_file(self._resolve_path(self.args.data.entity)),
            key=lambda x: x['id']
        )
        self.kb_entity = self.setup_dataset_for_entity(
            self._resolve_path(self.args.data.entity),
            self.raw_kb_entity
        )
        self.kb_id2entity = {
            raw_ent['id']: ent
            for raw_ent, ent in zip(self.raw_kb_entity, self.kb_entity)
        }

        self.train_data = self.setup_dataset_for_mention(
            self._resolve_path(self.args.data.train_file),
            _load_json_file(self._resolve_path(self.args.data.train_file))
        )
        self.val_data = self.setup_dataset_for_mention(
            self._resolve_path(self.args.data.dev_file),
            _load_json_file(self._resolve_path(self.args.data.dev_file))
        )
        self.test_data = self.setup_dataset_for_mention(
            self._resolve_path(self.args.data.test_file),
            _load_json_file(self._resolve_path(self.args.data.test_file))
        )

        print(f"\n数据统计:")
        print(f"  实体数量: {len(self.kb_entity)}")
        print(f"  训练样本: {len(self.train_data)}")
        print(f"  验证样本: {len(self.val_data)}")
        print(f"  测试样本: {len(self.test_data)}\n")

    def _resolve_path(self, path):
        if os.path.isabs(path) or not self.base_path:
            return path
        return os.path.join(self.base_path, path)

    def setup_dataset_for_entity(self, path, data):
        """处理实体数据"""
        pkl_path = path[:path.rfind('.')] + '.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                input_data = pickle.load(file)
            return input_data

        input_data = []
        for sample_dict in tqdm(data, desc='处理实体数据'):
            sample_type = sample_dict['type']
            if sample_type == 'entity':
                entity = unquote(sample_dict.pop('entity_name'))
                desc = sample_dict.pop('desc')
                input_text = entity + ' [SEP] ' + desc
                input_dict = self.tokenizer(
                    input_text,
                    padding='max_length',
                    max_length=self.args.data.text_max_length,
                    truncation=True
                )

            input_dict['img_list'] = sample_dict['image_list']
            input_dict['sample_type'] = 0 if sample_type == 'entity' else 1

            if 'answer' in sample_dict:
                input_dict['answer'] = self.qid2id[sample_dict['answer']]

            input_data.append(input_dict)

        with open(pkl_path, 'wb') as file:
            pickle.dump(input_data, file)

        return input_data

    def setup_dataset_for_mention(self, path, data):
        """处理提及数据"""
        pkl_path = path[:path.rfind('.')] + '.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as file:
                input_data = pickle.load(file)
            return input_data

        input_data = []
        for sample_dict in tqdm(data, desc='处理提及数据'):
            if sample_dict.get('answer') == 'nil':
                continue

            entity = unquote(sample_dict.pop('entities'))
            mention = unquote(sample_dict.pop('mentions'))
            text = sample_dict.pop('sentence')
            desc = sample_dict.pop('desc')

            input_text = mention + ' [SEP] ' + text + ' [SEP] ' + desc

            input_dict = self.tokenizer(
                input_text,
                padding='max_length',
                max_length=self.args.data.text_max_length,
                truncation=True
            )

            img_path = sample_dict['imgPath']
            input_dict['img_list'] = [img_path] if img_path != '' else []
            input_dict['sample_type'] = 1  # 1表示mention

            if 'answer' in sample_dict:
                input_dict['answer'] = self.qid2id[sample_dict['answer']]

            input_data.append(input_dict)

        with open(pkl_path, 'wb') as file:
            pickle.dump(input_data, file)

        return input_data

    def choose_image(self, sample_type, img_list, is_eval=False):
        """
        选择并加载图像
        Args:
            sample_type: 0=entity, 1=mention
            img_list: 图像路径列表
            is_eval: 是否为评估模式
        """
        if len(img_list) > 0:
            img_name = img_list[0] if is_eval else random.choice(img_list)

            if sample_type == 1:
                img_name = img_name.split('/')[-1].split('.')[0] + '.jpg'

            try:
                img_folder = (self._resolve_path(self.args.data.kb_img_folder)
                              if sample_type == 0
                              else self._resolve_path(self.args.data.mention_img_folder))
                img_path = os.path.join(img_folder, img_name)

                img = Image.open(img_path).resize((224, 224), Image.Resampling.LANCZOS)
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
            except Exception as e:
                pixel_values = torch.zeros((3, 224, 224))
        else:
            pixel_values = torch.zeros((3, 224, 224))

        return pixel_values

    def train_collator(self, samples):
        """训练数据批处理"""
        img_list, sample_type, input_dict_list = [], [], []
        pixel_values, gt_ent_id = [], []

        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))
            input_dict_list.append(sample)

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx]))

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)

        ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in gt_ent_id]
        ent_img_list, ent_type, ent_input_dict_list = [], [], []
        ent_pixel_values = []

        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))
            ent_type.append(ent_dict.pop('sample_type'))
            ent_input_dict_list.append(ent_dict)

        for idx, _ in enumerate(ent_input_dict_list):
            ent_pixel_values.append(self.choose_image(ent_type[idx], ent_img_list[idx]))

        ent_empty_img_flag = torch.tensor(
            [True if not len(img_list) else False for img_list in ent_img_list],
            dtype=torch.bool
        )

        ent_input_dict = self.tokenizer.pad(
            ent_input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        ent_input_dict['pixel_values'] = torch.stack(ent_pixel_values)
        ent_input_dict['empty_img_flag'] = ent_empty_img_flag

        for k, v in ent_input_dict.items():
            input_dict[f'ent_{k}'] = v

        return input_dict

    def eval_collator(self, samples):
        """评估数据批处理（仅处理mention）"""
        img_list, sample_type, input_dict_list = [], [], []
        pixel_values, gt_ent_id = [], []

        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))
            input_dict_list.append(sample)

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)
        input_dict['answer'] = torch.tensor(gt_ent_id, dtype=torch.long)

        return input_dict

    def entity_collator(self, samples):
        """实体数据批处理"""
        pixel_values, img_list, sample_type, input_dict_list = [], [], [], []

        for sample in samples:
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)

        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))

        input_dict = self.tokenizer.pad(
            input_dict_list,
            padding='max_length',
            max_length=self.args.data.text_max_length,
            return_tensors='pt'
        )
        input_dict['pixel_values'] = torch.stack(pixel_values)

        return input_dict

    def entity_dataloader(self):
        """实体数据加载器"""
        return DataLoader(
            self.kb_entity,
            batch_size=self.args.data.embed_update_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.entity_collator
        )

    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_data,
            batch_size=self.args.data.batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=True,
            collate_fn=self.train_collator
        )

    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_data,
            batch_size=self.args.data.eval_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.eval_collator
        )

    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_data,
            batch_size=self.args.data.eval_batch_size,
            num_workers=self.args.data.num_workers,
            shuffle=False,
            collate_fn=self.eval_collator
        )
