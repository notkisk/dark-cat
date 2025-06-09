import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import time

# --- Configuration ---
CHECKPOINT_PATH = "checkpoint.pth" # Make sure this file is in the same directory
IMAGE_PATH = "form.png" # <--- **CHANGE THIS TO YOUR IMAGE FILE PATH**
NUM_CLASSES = 16 # Must match the number of classes used during training

# RVL-CDIP Class Labels (assuming you used the standard RVL-CDIP dataset)
# You can verify these from your original code's print(id2label) output
id2label = {
    0: 'letter',
    1: 'form',
    2: 'email',
    3: 'handwritten',
    4: 'advertisement',
    5: 'scientific report',
    6: 'scientific publication',
    7: 'specification',
    8: 'file folder',
    9: 'news article',
    10: 'budget',
    11: 'invoice',
    12: 'presentation',
    13: 'questionnaire',
    14: 'resume',
    15: 'memo'
}


def inference(image_path):
    start_time = time.time()
    # --- Set Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define the Model Architecture ---
    # This MUST match the architecture used during training
    model = models.efficientnet_b0() # pretrained=False because we load our own weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)

    # --- Load the Saved Checkpoint ---
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit()

    try:
        # Load the entire dictionary saved during checkpointing
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded successfully from {CHECKPOINT_PATH}")

    except KeyError:
        print(f"Warning: 'model_state_dict' key not found in {CHECKPOINT_PATH}. Trying to load directly...")
        # Fallback in case the checkpoint only saved the state_dict directly
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
            print(f"Model weights loaded successfully (assuming checkpoint only contained state_dict).")
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            exit()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit()


    # --- Set Model to Evaluation Mode ---
    model.eval()

    # --- Define Image Transformations ---
    # These MUST match the validation/test transforms used during training
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")), # Ensure image is RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) # Same normalization as training
    ])

    # --- Load and Preprocess the Image ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        exit()

    try:
        image = Image.open(image_path)
        image = val_transform(image)
        # Add a batch dimension (models expect input in batches, even if it's just one image)
        image = image.unsqueeze(0)
        image = image.to(device) # Move image to the same device as the model
        print(f"Image loaded and preprocessed: {image_path}")

    except Exception as e:
        print(f"Error loading or processing image: {e}")
        exit()

    # --- Perform Inference ---
    print("Performing inference...")
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(image)

    # --- Get Prediction and Probabilities ---
    # The output tensor contains the scores for each class
    probabilities = torch.softmax(outputs, dim=1)[0] # Get probabilities for the single image in the batch
    confidence, predicted_index = torch.max(probabilities, dim=0) # Get the class with the highest probability

    predicted_label = id2label[predicted_index.item()] # Look up the class name using the index

    # --- Print Results ---
    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence.item():.4f}")

    # Optional: Print top N probabilities
    N = 5
    top_p, top_class_indices = probabilities.topk(N, dim=0)
    print(f"\nTop {N} predictions:")
    for i in range(N):
        print(f"  {id2label[top_class_indices[i].item()]}: {top_p[i].item():.4f}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Dark Cat')
    
    parser.add_argument('img_path', help='Image Path duh!')
    
    args = parser.parse_args()
    
    inference(args.img_path)


if __name__ == "__main__":
    main()