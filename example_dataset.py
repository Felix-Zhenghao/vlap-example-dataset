import os

import torch
import torchvision
import numpy as np
from PIL import Image

class ExampleDataset(torch.utils.data.Dataset):
    """
    - Dataset Format
    ```
    example_dataset/
    ├── data_0/
    │   ├── input_images/
    │   │   ├── 0.jpg (wrist view at t-1)
    │   │   ├── 1.jpg (wrist view at t)
    │   │   ├── 2.jpg (agent view at t-1)
    │   │   └── 3.jpg (agent view at t)
    │   ├── text.txt (task description)
    │   └── target.jpg (agentview at t+8)
    ```
    
    - Output Collated Batch
        - pixel_values: dict
            - dino: torch.Tensor (bsz, history_img_seq_len, 3, 224, 224)
            - siglip: torch.Tensor (bsz, history_img_seq_len, 3, 224, 224)
        - input_ids: torch.Tensor (bsz, seq_len)
        - [dummy] attention_mask: torch.Tensor (bsz, seq_len)
        - [dummy] labels: torch.Tensor (bsz, seq_len)
        - goal_image: torch.Tensor (bsz, 3, 224, 224)
        - [dummy] actions: torch.Tensor (bsz, action_horizon, action_dim)
        - [dummy] proprios: torch.Tensor (bsz, history_horizon, 7) # if use wrist, history_img_seq_len = 2 * history_horizon
        - [dummy] future_wrist_view: torch.Tensor (bsz, 3, 224, 224)
        - [dummy] language_planning_input_ids: not involved in the example dataset
        - [dummy] language_planning_attention_mask: not involved in the example dataset
        - [dummy] language_planning_labels: not involved in the example dataset
    """

    def __init__(self, root_dir, image_transform=None, text_tokenizer=None, resize_size=(224, 224)):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset. For instance "/data/czh/Download/vla_planning/example_dataset".
            image_transform (callable, optional): Transform to be applied on the *history* images.
            text_tokenizer (callable, optional): Tokenizer for the text data.
            resize_size (tuple, optional): Size to resize the images to. Default is (224, 224).
        """
        self.root_dir = root_dir
        self.samples = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("data_")]
        self.resize = torchvision.transforms.Resize(resize_size)
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        
        # Load input images
        input_images = []
        input_dir = os.path.join(sample_path, 'input_images')
        for i in range(4):  # Load 4 images
            img_path = os.path.join(input_dir, f'{i}.jpg')
            image = Image.open(img_path).convert('RGB')
            input_images.append(self.resize(image))
        pixel_values = self.image_transform(input_images)
        
        # Load text
        with open(os.path.join(sample_path, 'text.txt'), 'r') as f:
            task_description = f.read().strip()
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{task_description}"},
        ]
        texts = self.text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True,).input_ids.squeeze()

        # Load target image
        target = Image.open(os.path.join(sample_path, 'target.jpg')).convert('RGB')
        target = torch.tensor(np.asarray(self.resize(target))).to(torch.float32).div(255/2).sub(1).permute(2, 0, 1)
        
        return {
            "dino": pixel_values["dino"].to("cuda"),
            "siglip": pixel_values["siglip"].to("cuda"),
            "goal_image": target.to("cuda"),
            "input_ids": llm_inputs.to("cuda"),
            "attention_mask": torch.ones_like(llm_inputs, dtype=torch.bool).to("cuda"),
            "labels": torch.full_like(llm_inputs, -100, dtype=torch.int64).to("cuda"),
            "actions": torch.zeros((1, 7), dtype=torch.float32).to("cuda"),  # Dummy action
            "proprios": torch.zeros((1, 9), dtype=torch.float32).to("cuda"),  # Dummy proprio
            "future_wrist_view": torch.zeros((3, 224, 224), dtype=torch.float32).to("cuda"),  # Dummy future wrist view
        }
        
    def collate_fn(self, batch):
        # Collate function to combine multiple samples into a batch
        collated_batch = torch.utils.data.default_collate(batch)
        collated_batch["pixel_values"] = {
            "dino": collated_batch["dino"],
            "siglip": collated_batch["siglip"],
        }
        collated_batch.pop("dino")
        collated_batch.pop("siglip")
        
        return collated_batch
