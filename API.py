from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import cnn
import utils

app = FastAPI()

# Load model
model = cnn.init_cnn(device="cpu", input_channel=1, out_channels=[100, 100], output_size=2)
model = utils.load_model(model, device="cpu")
model.eval()

# Define the image transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Load the uploaded image
    image = Image.open(io.BytesIO(await file.read()))

    # Apply the preprocessing
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the image through the model
    with torch.no_grad():
        output = model(image)

    # Convert output to a list for JSON serialization
    predictions = output.squeeze().tolist()

    # Return the predictions as a JSON response
    return JSONResponse(content={"perspective_score_hood": predictions[0],
                                 "perspective_score_backdoor_left": predictions[1]})

