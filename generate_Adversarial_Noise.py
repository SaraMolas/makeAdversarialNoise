
def generate_Adversarial_Noise(image_path, target_string): 

    # 1. Set parameters
    epsilon = 0.01 # noise added to the adversarial image
    lr = 0.01 # learning rate for the optimizer of delta vector
    iterations = 10 # number of iterations performed during optimization

    # 2. Import necessary libraries
    import os
    import requests
    from PIL import Image

    import torch
    dev = None
    if torch.cuda.is_available():
    dev = torch.device('cuda')
    else:
    dev = torch.device("cpu")
    print(f"Using device '{dev}'")

    from torchvision import models, transforms
    from torch import nn
    import torch.nn.functional as F

    import matplotlib.pyplot as plt
    import numpy as np
    import copy
    import json

    # 3. Get list of target classes
    imagenet_classes_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    imagenet_classes = requests.get(imagenet_classes_url).json()

    
    # 4. Create basic function for plotting

    def plot_image_with_label(image, output): 
        
        # function to plot the input image of the model and the label assigned
        # image = input image, expected to be in PIL image format
        # output = output from the model, containing the probabilities for all the classes
        
        # Convert ResNet50 output to probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get index of most likely class
        index = torch.argmax(probabilities, dim=1).item()

        # Get the probability of the most likely class
        p = probabilities[0, index].item()
        
        # Create figure
        plt.imshow(image) # plot image
        plt.title(f"Predicted class: {imagenet_classes[index]} with p = {p:.3}") # indicate predicted label and p
        plt.axis('off') # remove axis
        plt.show() # show figure

    # 5. Upload input image and apply transformations

    # Load image
    image = Image.open(image_path).convert("RGB") 

    # Function to convert image to the size used to train ResNet50 (224 x 224 pixels)
    input_transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224)])

    # Function used later to transform image to tensor and from tensor to image
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Resize input image and convert to tensor
    image = input_transform(image)
    img_tensor = to_tensor(image)

    # 6. Load pre-trained model 
    resnet50 = models.resnet50(pretrained=True).to(dev)
    resnet50.eval()

    # Add a normalization layer to the model before the pre-existing input layer
    normalization_layer = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )

    # Make a copy of the model just in case
    model = copy.deepcopy(resnet50)

    # Prepend the transformation
    model = nn.Sequential (normalization_layer, model)

    # 7. Predict label of original image + plot

    # Input tensor image to the model and make prediction
    output = model(img_tensor.unsqueeze(0)) # we need to add a fourth dimension to the tensor

    # Plot results
    plot_image_with_label(image, output)

    # 8. Define a module for a vector delta = noise

    # define a module for delta, the noise
    class Adversarial_modification(nn.Module):
        
        def __init__(self, input_tensor, epsilon):
            
            super(Adversarial_modification, self).__init__()
            self.epsilon = epsilon
            
            # Initialize delta as a learnable parameter
            self.delta = nn.Parameter(torch.zeros_like(input_tensor, requires_grad=True)).to(dev)

        def forward(self, input_tensor):
            
            # Apply noise to the input tensor
            adv_tensor = input_tensor + self.delta
            
            # Clamp the modification so it stays within the valid range of pixel values 
            adv_tensor = torch.clamp(adv_tensor, 0, 1)  
            
            # Optionally, constrain the delta norm
            self.delta.data = torch.clamp(self.delta.data, -self.epsilon, self.epsilon)
            return adv_tensor
        
    # 9. Perform adversarial attack using FGSM and obtain modified image

    # Find index of target label
    target_id = imagenet_classes.index(target_string) # get index of target class
    target_tensor = torch.tensor(target_id).unsqueeze(0) # convert index into tensor

    # Set up autograd
    img_tensor.requires_grad = True

    # Initialize optimizer
    delta = Adversarial_modification(img_tensor, epsilon)
    optimizer = torch.optim.Adam(delta.parameters(), lr=lr)

    # iterate a few times
    for step in range(iterations):
        
        # compute gradient
        optimizer.zero_grad()
        output = model(delta.forward(img_tensor.unsqueeze(0)))
        loss = F.cross_entropy(output, target_tensor)
        
        loss.backward()
        
        # optimize
        optimizer.step()

    # Extract final adversarial tensor    
    adv_tensor = delta.forward(img_tensor)

    # 10. Predict label of modified image + plot
    output = model(adv_tensor.unsqueeze(0)) # we need to add a fourth dimension to the tensor

    # convert adversarial tensor to image for plotting
    adv_image = to_pil(adv_tensor)

    # Plot image with adversarial noise and predicted label
    plot_image_with_label(adv_image, output)

    # 11. Save modified image

    # adversarial image filepath
    current_path = os.getcwd()
    adv_path = current_path + '/adversarial_image.jpg'

    # Save the image
    adv_image.save(adv_path)

    return f"image saved as {adv_path}"
    