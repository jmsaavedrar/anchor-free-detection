import torch
from torchvision import transforms
from PIL import Image
from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor

if __name__ == '__main__':            
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # defining the image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
        ])
    #load de model 
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)    
    # return_nodes = ['blocks.11.mlp.fc2']
    # model2 = create_feature_extractor(model, return_nodes = return_nodes)


    image_path = "images/example_2.jpg"
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # add an extra dimension for batch
    #Pasamos la imagen por el modelo    
    with torch.no_grad():        
        features = model.forward_features(image)
        patch_tokens = features['x_norm_patchtokens']
        print(patch_tokens.shape)
        
        # dim = patch_tokens.shape
        # print('dim = {}'.format(dim))   
    # dim = features.shape                                
    # print('dim = {}'.format(dim))   
    # for name, module in model.named_modules():
    #     print(name, module)    
    # #summary(model, (3, 224, 224))
    #print(features)
