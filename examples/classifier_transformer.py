import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from transformers import VisionTransformer

if __name__ == "__main__":

    classifier_transformer = VisionTransformer(
        embedding_dimension = 768, # It is the dimension that all tokens will receive to capture more complex semantic relationships between words.
        number_heads = 6, # Number heads used in the self-attention process
        feed_forward_dimension = 2048, # Dimmension number of each output of coder or decoder
        dropout = 0.1, # Percentage of neurons that will be turned off during training to avoid overfitting in encoders and decoders
        image_size = 16, # Size of the images in the training set. The images must have all the same size
        input_channels = 1, # The input channel can be 1 (black and white) or 3 (red, green, blue)
        patch_size = 16, # The pixel size of each patch. A patch is a fraction of the image. If the image have 256 pixels, and we have patches of size 16 pixels, we will have 16 patches
        num_classes = 10, # the number of classes into which the images in this training set could be divided,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path = '/saved/classifier_tranformer'
    )

    transform = transforms.Compose([
        transforms.Resize((16, 16)),  # The input channel can be 1 (black and white) or 3 (red, green, blue)
        transforms.ToTensor(),
        transforms.Normalize((0.5, ),(0.5, )),  # Normalizes pixel values ​​to be in the range [-1, 1]
    ])

    train_dataset = datasets.USPS(root="./data", train=True, download=True, transform=transform)

    """The transformer created above is trained in a 2000 epoch cycle with all Spanish-English phrases"""
    classifier_transformer.train(train_dataset, num_epochs=2000)

    classifier_transformer(train_dataset)