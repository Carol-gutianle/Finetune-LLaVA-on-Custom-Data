import os
import json
import torch
from PIL import Image
import transformers
from copy import deepcopy
from typing import Sequence, Dict
from torch.utils.data import Dataset
from dataclasses import dataclass

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

def load_data_from_json(data_path):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def preprocess_multimodal(
    sources: Sequence[str]
) -> Dict:
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
    return sources

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
):
    conversations = []
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

    targets = deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = get_tokenize_len([s["value"] for s in source])
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class Dataset(Dataset):
    def __init__(self, data, tokenizer, processor):
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        sources = self.data[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Wrong list"
        if 'image' in sources[0]:
            image_file = self.data[i]['image']
            image_folder = '/mnt/petrelfs/gutianle/MLLMGuard'
            processor = self.processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = processor(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                deepcopy([e['conversations'] for e in sources])
            )
        data_dict = preprocess(
            sources,
            self.tokenizer
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids = data_dict['input_ids'][0],
                labels = data_dict['labels'][0]
            )
        if 'image' in self.data[i]:
            data_dict['image'] = image
        return data_dict
    
class RawDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        sample = self.data[i]
        return sample
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch