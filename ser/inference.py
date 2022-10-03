from ser.transforms import transforms, normalize
import torch
   
def setup_inference(test_dataloader, label):
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images, labels


def infer1(model, images, params, label):
    print("Inference is run for the model\n -  {params.name} using the following hyperparameters:")
    print("Epochs : {params.epoch}")
    print("Batch size: {params.batch_size}")
    print("Learning Rate: {params.learning_rate}")
    print("The image being classified is a {label}")
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")
    print(f"The certainty of this is {certainty * 100}")

def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "