import sys
import torch
import clip
import getFiles
import numpy
from PIL import Image

device1 = "cuda" if torch.cuda.is_available() else "cpu"

catagories_prompts = ["a diagram", "a dog", "a cat", "a cat girl"]

def classfiy_image(image_path):
    
    model, preprocess = clip.load("ViT-B/32", device=device1)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device1)
    text = clip.tokenize(catagories_prompts).to(device1)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        
    
        return probs

def compare_probs(probs):
    first_prob = probs[0]
    prob_index_array = numpy.where(numpy.isclose(first_prob, (max(first_prob)))) # numpy.where returns the index of the specified element
    prob_index = prob_index_array[0][0] # prob[0] is still an element of a numpy array
    
    if prob_index == 0:
        return "diagram"
    elif prob_index == 1:
        return "dog"
    elif prob_index == 2:
        return "cat"
    else:
        return "cat girl"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python getFiles.py <dir>")
        sys.exit(1)
    rawdir = sys.argv[1]
    allfiles = getFiles.get_first_level_file(rawdir)
    for f in allfiles:
        if getFiles.is_valid_image(f):
            probs = classfiy_image(f)
            # classfiy_result = compare_probs(probs)
            print(probs[0])
            print(f, "is a picture of", compare_probs(probs)) # prints: If the first number is the biggest, it's a diagram. If the second number is the biggest, it's a dog. If the third number is the biggest, it's a cat.
             