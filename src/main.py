

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset_processing.images_dataset import ImagesDataset

batch_size = 4

def init():

    
    csv_file_path = "/home/kamal/Downloads/archive/train.csv"
    folder_path = "/home/kamal/Downloads/archive/train"

    # DATA AUGMENTATION
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])



    images_dataset = ImagesDataset(folder_path, csv_file_path, transform)

    dataloader = DataLoader(images_dataset, batch_size=batch_size, shuffle=True)

    for images, labels in dataloader:
        print(f"Batch of images and labels:")
        
        for i in range(len(images)):
            image = images[i]
            target = labels[i]
            
            print(image)
            print(f"    Label: {target.item()}")  

 
    return ""

if __name__ == "__main__":
    init()