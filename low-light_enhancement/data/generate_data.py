import os
import torch
from PIL import Image
from torchvision import transforms
import requests

class ImagePromptDataset:
    def __init__(self, image_dir, transform=None, device='cuda'):
        """
        Initialize the dataset.

        Parameters:
        - image_dir: The directory containing the images.
        - llm_api_url: The URL of the LLM API for generating prompts.
        - transform: Optional torchvision transforms to apply to the images.
        - device: The device to use ('cuda' or 'cpu').
        """
        self.image_dir = image_dir
        self.device = device
        self.transform = transform if transform else transforms.ToTensor()
        self.image_paths = self._load_image_paths()
        self.data = self._create_data_dict()

    def _load_image_paths(self):
        """Load all image paths from the specified directory."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        return [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(image_extensions)]

    def _generate_prompt(self, image_path):
        """Generate a prompt for a given image using the LLM API."""
        from openai import OpenAI
        base_url = "https://openkey.cloud/v1"
        api_key = "sk-xmY83ICHAbw95y8RBeF0B92143154dF78b0bC762912d78D8"
        client = OpenAI(base_url=base_url, api_key=api_key)

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
        )
        return completion.choices[0].message.content

    def _create_data_dict(self):
        """Create a dictionary with image tensors and corresponding prompts."""
        data_dict = {}
        for image_path in self.image_paths:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).to(self.device)
            prompt = self._generate_prompt(image_path)
            data_dict[image_tensor] = prompt
        return data_dict

    def __getitem__(self, index):
        """Get a single data item."""
        image_path = self.image_paths[index]
        image_tensor = self.transform(Image.open(image_path).convert('RGB')).to(self.device)
        prompt = self._generate_prompt(image_path)
        return image_tensor, prompt

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def get_data(self):
        """Return the complete data dictionary."""
        return self.data

# Example usage
if __name__ == '__main__':
    image_directory = '/path/to/your/images'

    dataset = ImagePromptDataset(image_directory)
    data = dataset.get_data()

    # Example of accessing the data
    for image_tensor, prompt in data.items():
        print(f"Image Tensor: {image_tensor.shape}, Prompt: {prompt}")
