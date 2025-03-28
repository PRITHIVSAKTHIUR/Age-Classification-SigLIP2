![AAAAAAAA.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/hWrRztXlZ0j87BEajVNtA.png)

# **Age-Classification-SigLIP2**  

> **Age-Classification-SigLIP2** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to predict the age group of a person from an image using the **SiglipForImageClassification** architecture.  

```py
Classification Report:
                  precision    recall  f1-score   support

      Child 0-12     0.9744    0.9562    0.9652      2193
  Teenager 13-20     0.8675    0.7032    0.7768      1779
     Adult 21-44     0.9053    0.9769    0.9397      9999
Middle Age 45-64     0.9059    0.8317    0.8672      3785
        Aged 65+     0.9144    0.8397    0.8755      1260

        accuracy                         0.9109     19016
       macro avg     0.9135    0.8615    0.8849     19016
    weighted avg     0.9105    0.9109    0.9087     19016
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/rgfZs4duAb09vRvFmO3Qy.png)



The model categorizes images into five age groups:  
- **Class 0:** "Child 0-12"  
- **Class 1:** "Teenager 13-20"  
- **Class 2:** "Adult 21-44"  
- **Class 3:** "Middle Age 45-64"  
- **Class 4:** "Aged 65+"  

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```  

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Age-Classification-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def age_classification(image):
    """Predicts the age group of a person from an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Child 0-12", 
        "1": "Teenager 13-20", 
        "2": "Adult 21-44", 
        "3": "Middle Age 45-64", 
        "4": "Aged 65+"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=age_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Age Group Classification",
    description="Upload an image to predict the person's age group."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```
# **Sample Inference:**  

![Screenshot 2025-03-28 at 12-25-46 Age Group Classification.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ARlNhc-ZxqfBntu-SkIVH.png)

![Screenshot 2025-03-28 at 12-36-49 Age Group Classification.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/tvZ2VMoaQqNKdIx39DrTe.png)

# **Intended Use:**  

The **Age-Classification-SigLIP2** model is designed to classify images into five age categories. Potential use cases include:  

- **Demographic Analysis:** Helping businesses and researchers analyze age distribution.  
- **Health & Fitness Applications:** Assisting in age-based health recommendations.  
- **Security & Access Control:** Implementing age verification in digital systems.  
- **Retail & Marketing:** Enhancing personalized customer experiences.  
- **Forensics & Surveillance:** Aiding in age estimation for security purposes.  
