import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImagesDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):

        self.image_folder = image_folder
        self.csv_file = csv_file
        self.transform = transform
        
        self.data = pd.read_csv(csv_file)
        
        self.image_names = self.data['image_name'].tolist()

        self.labels = self.data['target'].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        label = self.labels[idx]
        
        image_path = os.path.join(self.image_folder, image_name)

        try:
            image = Image.open(image_path+".jpg").convert('RGB')  
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)

        return image, label  

    def _getLabelByImageName(self, image_name):
        
        try:
            idx = self.image_names.index(image_name)
            return self.labels[idx]
        except ValueError:
            print(f"Image '{image_name}' not found in the dataset.")
            return None