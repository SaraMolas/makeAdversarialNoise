# makeAdversarialNoise
Targeted adversarial attack on an image classifier (ResNet50)

This repository demonstrates the implementation of an Adam optimizer-based approach for generating adversarial perturbations in ResNet50, an image classification model. This approach focuses on enhancing the effectiveness of targeted adversarial attacks by employing iterative optimization.

# How it works 
Initialization: Perturbation (delta) is initialized to zero and iteratively updated.
Optimization: Gradients of the model loss with respect to the input are computed. The Adam optimizer updates the perturbation (delta) over multiple steps.
Constraint Enforcement: Perturbations are clamped to the valid pixel range. 

# Files 
- **generate_Adversarial_Noise.py**: full script to run the code at once. It takes as input the directory of the image and the target class. Generates the adversarial image and saves it within the same folder in which this file is located. The function returns a message indicating the directory where the image is saved. 
- **TargetedAdversarialAttack.ipynb**: Jupyter notebook showcasing an example of this approach, where adversarial noise was introduced in an image of a Labrador Retriever to get the model to classify it as a banana. 

# Input 
- Image: string of filepath to image, picture provided by the user. 
- Target: string, desired predicted class. 

# Output 
- Adv_image: original image with added adversarial noise. 

# Things to do: 
- Dependency management: Create a requirements.txt file
- Code structure: organize final code into models
- Provide sample data for testing
