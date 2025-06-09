# 🐾 Dark Cat – Document Image Classifier

**Dark Cat** is a PyTorch-based document image classification model trained to distinguish between **16 different types of documents**. It is built for practical document sorting and content-based image analysis.

---

## 📚 Dataset

- **Dataset**: [RVL-CDIP]([https://huggingface.co/datasets/aharley/rvl_cdip]))
- **Total images**: 400,000 document images
- **Classes**: 16
- **Image format**: Grayscale document images, each sized 224×224 during training

### Dataset split:
- **Training set**: 320,000 images  
- **Validation set**: 40,000 images  
- **Test set**: 40,000 images

---

## 📦 Files

- `dark-cat.pth`: Saved model weights (`state_dict` only).
- `checkpoint.pth`: Full training checkpoint containing:
  - Model weights
  - Optimizer state
  - Epoch number
  - Training/validation accuracy and loss

---

## 🧠 Classes (Index → Label)

0 → letter
1 → form
2 → email
3 → handwritten
4 → advertisement
5 → scientific report
6 → scientific publication
7 → specification
8 → file folder
9 → news article
10 → budget
11 → invoice
12 → presentation
13 → questionnaire
14 → resume
15 → memo


📊 Training Summary
Dataset: RVL-CDIP (400,000 images)

Epochs: 5

Training Accuracy: 93.07%

Validation Accuracy: 91.07%

Final Loss: 1115.5758
