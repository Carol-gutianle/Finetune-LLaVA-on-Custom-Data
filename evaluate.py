import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from PIL import Image

from dataloader import gather_data, RawDataset
from tqdm import tqdm

class Llava:
    
    def __init__(self, model_name_or_path, **kwargs):
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained("/mnt/cache/gutianle/llava-1.5-7b-hf")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f'Evaluating {model_name_or_path}...')
        
    def evaluate(self, prompt, filepath):
        image = Image.open(filepath)
        prompt = f"<image>\nUSER: {prompt} \nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
        generate_ids = self.model.generate(**inputs, max_length=128)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # post process
        replace_text = 'ASSISTANT: '
        output = output[output.find(replace_text) + len(replace_text):]
        return output
    
    def convert_to_prompt(self, sample, mode='raw'):
        conversations = sample['conversations']
        image = sample['image']
        image = Image.open(os.path.join('./imgs',image)).convert('RGB')
        answer = ''
        if mode == 'raw':
            prompt = ''
            for conv in conversations:
                if conv['role'] == 'user':
                    for i, item in enumerate(conv['content']):
                        if item['type'] == 'text':
                            prompt = f"<image>\nUSER: {conv['content'][i]['text']} \nASSISTANT:"
                elif conv['role'] == 'evaluator':
                    answer = conv['content'][0]['text']
        elif mode == 'tuned':
            prompt = self.processor.apply_chat_template(
                conversations,
                tokenize = False
            )
            prompt = prompt[:prompt.find('EVALUATOR')] + ' EVALUATOR: '
            for conv in conversations:
                if conv['role'] == 'evaluator':
                    answer = conv['content'][0]['text']
        return prompt, image, answer
    
    def run(self, mode):
        for dim in ['non-existent']:
            test_dataset = gather_data('test', dim)
            test_dataset = RawDataset(test_dataset)
            total = len(test_dataset)
            cnt = 0
            print(f'当前的维度:{dim}')
            with tqdm(test_dataset) as tbar:
                for sample in tbar:
                    try:
                        prompt, image, answer = self.convert_to_prompt(sample, mode)
                        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
                        generate_ids = self.model.generate(**inputs, max_new_tokens=512)
                        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        # post process
                        if mode == 'raw':
                            replace_text = 'ASSISTANT: '
                            output = output[output.find(replace_text) + len(replace_text):]
                        elif mode == 'tuned':
                            replace_text = 'EVALUATOR: '
                            output = output[output.find(replace_text) + len(replace_text): output.find(replace_text) + len(replace_text) + 10]
                        if answer in output:
                            cnt += 1
                    except:
                        cnt += 1
                        output = 'error'
                    tbar.set_postfix({'acc': cnt / total, 'output': output})
            print(f'当前的维度:{dim}, 准确率:{cnt / total}')
        
mllm = Llava('/mnt/cachenew/gutianle/llavarank')
mllm.run('tuned')
        