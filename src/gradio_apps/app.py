import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def preprocess_ease(input_image):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])  
    print(type(input_image))
    input_image2 = image_transform(input_image)

    return input_image2


# Load the classes
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load the pre-trained model and the saved weights
def create_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))  # Adjust for 6 classes
    model.load_state_dict(torch.load("./ResNet18/model_resnet_checkpoint_1915.pth"))
    
    return model

# Predict function
def predict(input_image):
    X = preprocess_ease(input_image)
    X = X.squeeze()
    model = create_model()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_input = torch.unsqueeze(X, dim=0)
        outputs = model(X_input)
        probabilities = torch.softmax(outputs[0], dim=0)
        pred_result = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        print(pred_result)
    return pred_result

title = "Welcome on your first trash classification app!"

head = (
  "<center>"
  """This app utilizes a pretrained model, ResNet, that has been repurposed to classify one of six types: cardboard, glass, metal, papers, plastics, trash. 
   Feel free to use this for prediction. Cheers."""
  "</center>"
)

ref = "Took inspiration from [here](https://gradio.app/image_classification_in_tensorflow/)."

def main():
    interface = gr.Interface(fn=predict, 
                             inputs=gr.Image(type="pil"),
                             outputs=gr.Label(num_top_classes=6),
                             examples=[
                                 "cardboard.png", "glass.png",
                                 "metal.png", "paper.png",
                                 "plastic.png", "trash.png"
                             ],
                             title=title, 
                             description=head, 
                             article=ref,
                             allow_flagging='never')
    return interface.launch()
       

if __name__ == '__main__':
    main()