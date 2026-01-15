import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torchvision.models.vision_transformer import vit_b_16
from torch import nn
from torchvision.models import vit_b_16
from torchvision.models import vit_b_16, ViT_B_16_Weights
import faiss
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import gradio as gr

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # <-- add 

# 4
# preprocess images
transform = transforms.Compose([
    transforms.Resize((224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) # Normalization values of ImageNet 
])

def preprocess_image(image_path):
    img= Image.open(image_path).convert("RGB")
    original_size = img.size #width, height
    img_tensor = transform(img)
    metadata= {
        "path":str(image_path),
        "original_size": original_size,
        "processed_shape": img_tensor.shape,
        "processed_dtype": str(img_tensor.dtype),
        "processed_device": str(img_tensor.device)       
    }
    return img_tensor, metadata
# accessing all images in the directory
def preprocess_images_in_folder(folder_path):
    print("inside the  preprocess_images_in_folder function")
    folder= Path(folder_path)
    # adjust the extensions as needed
    image_paths= list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.webp"))
    if not image_paths:
        print(f"NO images found in: {folder_path}")
        
    all_tensors = []
    all_metadata = []
    # print(f"Found {len(image_paths)} images in {folder_path}")
    for img_path in image_paths:
        # print("inside the for loop")
        tensor, metadata = preprocess_image(img_path)
        all_tensors.append(tensor)
        all_metadata.append(metadata)
        
    print(f"Completed preprocessing {len(all_tensors)} images in {folder_path}")    
    return all_tensors, all_metadata
#after completion of this function it returns tensors of all images and metadata of all images
# Tensors used as input to vit , metadata used for displaying the results
# ==================================================
#  1
# load the saved model
def load_saved_model(path="vit_b16_feature_extractor.pth"):
    
    # print(f"loading the saved model from {path}... load_saved_model function")
    # model = vit_b_16(pretrained=False)
    model = vit_b_16(weights = None)  # Load the pre-trained weights  
    model.heads=nn.Identity()  # Replace classification head with identity
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()  # Set to eval mode
    # print("Model loaded successfully from the file. load_saved_model function")
    return model

# ==============================
# 2 3
# Load FAISS Index and Metadata
def load_faiss_index(index_path="faiss_ivf.index"):
    print(f"Loading FAISS index from {index_path}... load_faiss_index function")
    return faiss.read_index(index_path)


def load_metadata(path="metadata.json"):
    print(f"Loading metadata from {path}... load_metadata function")
    with open(path, "r") as f:
        return json.load(f)
    
#=================================================== 
#  5
# function  generate_gradcam
def generate_gradcam_vit(model, input_tensor, target_layer="encoder.ln"):
    print("this is the GRAD_CAM function")
    print(f"Generating Grad-CAM for the input tensor using {target_layer} layer")

    # Container for gradients and activations
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        print("Forward hook triggered")
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        print("Backward hook triggered")
        gradients['value'] = grad_out[0].detach()

    # Get the layer from the model using the layer name
    try:
        target_module = dict([*model.named_modules()])[target_layer]
    except KeyError:
        raise ValueError(f"Layer {target_layer} not found in the model.")

    # Register the hooks
    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_backward_hook(backward_hook)

    # Forward pass
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor.requires_grad_(True)
    output = model(input_tensor)  # shape: [1, 768]
    print(f"Output shape after forward pass: {output.shape}")

    # Create dummy target (pseudo target)
    target = output[0].mean()
    print(f"Target value for backward pass: {target.item()}")
    target.backward()
    print("Backward pass completed")

    # Check if hooks worked
    if 'value' not in activations or 'value' not in gradients:
        forward_handle.remove()
        backward_handle.remove()
        raise RuntimeError("Hooks did not fire. Ensure correct layer name is used.")

    # Get activations and gradients
    grads = gradients['value'][0]  # [tokens, channels]
    acts = activations['value'][0]  # [tokens, channels]

    weights = grads.mean(dim=0)  # [channels]
    print(f"Weights shape after global average pooling: {weights.shape}")

    cam = torch.matmul(acts, weights)  # [tokens]
    print(f"CAM shape after weighted combination: {cam.shape}")

    # Normalize and reshape
    cam =cam[1:]  # Exclude the CLS token
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.reshape(1, int(cam.shape[0]**0.5), int(cam.shape[0]**0.5))  # [1, H, W]

    # Upsample to input size
    cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = cam.squeeze().detach().cpu().numpy()

    # Clean up
    forward_handle.remove()
    backward_handle.remove()

    print("Grad-CAM generation completed")
    return heatmap
    # end 
    
# ===================================================
#  6
def overlay_heatmap_on_image(image_path, heatmap, alpha=0.5):
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to match heatmap size
    heatmap = cv2.resize(heatmap, (224, 224))  # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)  # Normalize heatmap to [0, 255]
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap
    overlayed = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)  # Overlay heatmap on image
    return overlayed
    # end
    
# ========================================================
#  7
# preprocess query image
def preprocess_query_image(image_path):
    print(f"Preprocessing of query image")
    img_tensor, _ = preprocess_image(image_path)
    return img_tensor

# =======================================================
#  8
# extract features from query image
# def extract_query_features(model,query_tensor):
#     print(f"Extracting features from query image")
#     model.eval()
#     with torch.no_grad():
#         query_tensor = query_tensor.unsqueeze(0)  # Add batch dimension
#         feature = model(query_tensor)  # Output shape: [1, 768]
#         feature_np = feature.squeeze(0).cpu().numpy().astype("float32")  # Remove batch dimension [768]
#         print(f"type(feature_np), feature_np.shape, feature_np.dtype: {type(feature_np)}, {feature_np.shape}, {feature_np.dtype}")
#         print(f"Extracted feature shape: {feature_np.shape}")  # Should be [768]
#         return feature_np

# Updated version of extract_query_features
def extract_query_features(model, query_tensor):
    print(f"Extracting features from query image...")
    model.eval()
    with torch.no_grad():
        query_tensor = query_tensor.unsqueeze(0)
        feature = model(query_tensor)
        feature_np = feature.squeeze(0).cpu().numpy().astype("float32")

        # Normalize to unit length (L2) for cosine similarity
        faiss.normalize_L2(feature_np.reshape(1, -1))

        print(f"Extracted normalized feature shape: {feature_np.shape}")
        return feature_np
# =========================================================

#  9
# k is the number of nearest neighbors to search for
# top_k is the number of results to return
# nprobe is the number of clusters to search in FAISS

# def search_faiss_index(query_feature, index, metadata, top_k=5, nprobe=2):
#     index.nprobe = nprobe
#     distances, indices = index.search(np.array([query_feature]), top_k)

#     results = []  # ✅ This is correctly a list

#     for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
#         if idx < len(metadata):
#             meta = metadata[idx]
#             similarity_score = 1 / (1 + distance + 1e-8)  # Avoid division by zero 
#             # there can be question on the similarity score , division by zero,if similarity is 100
#             result = {
#                 "rank": i + 1,
#                 "image_name": Path(meta["path"]).name,
#                 "image_path": meta["path"],
#                 "distance": float(distance),
#                 "similarity_score": float(similarity_score)
#             }
#             results.append(result)  # ✅ Appending to the list

#     print(f"Search completed. Found {len(results)} results.")
#     return results
# FOR COSINE SIMILARITY
def search_faiss_index(query_feature, index, metadata, top_k, nprobe=2):
    index.nprobe = min(nprobe,index.nlist ) # 🔁 Increase nprobe for better recall
    
    # Normalize query for cosine similarity
    query_feature = query_feature.astype('float32') # Ensure query feature is float32 for FAISS compatibility
    faiss.normalize_L2(query_feature.reshape(1, -1))  # Normalize to unit length for cosine similarity
    
    distances, indices = index.search(np.array([query_feature]), top_k)

    results = []
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx == -1 or idx >= len(metadata) or np.isnan(score) or np.isinf(score):
            print(f"Warning: Invalid result at rank {i+1}, skipping...")
            continue
        meta = metadata[idx]
        similarity_score = max(0.0, min(1.0, float(score)))

        # similarity_score =max(0.0, float(score))  # Cosine similarity ∈ [0, 1]
        # similarity_score= score
        distance = 1.0 - similarity_score  # Optional: interpret as cosine distance
        result = {
            "rank": len(results) + 1,  # Correct rank after filtering
            "image_name": Path(meta["path"]).name,
            "image_path": meta["path"],
            "similarity_score": similarity_score,
            "distance": distance
        
        }
        results.append(result)
    print(f"Search completed. Found {len(results)} results.")
    return results
# ===========================================


# ----------------------------------------------------------------------------------------------
# Load model, FAISS index, and metadata 
vit_model =load_saved_model("vit_b16_feature_extractor.pth")  # 1
faiss_index = load_faiss_index("vit_ivf.index") # 2
metadata = load_metadata("vit_metadata.json") # 3


# convert overlay results into images
def get_overlayed_gradcam_image(image_path, model):
    tensor,_ = preprocess_image(image_path) # 4
    heatmap = generate_gradcam_vit(model,tensor) # 5
    overlayed = overlay_heatmap_on_image(image_path, heatmap) # 6
    return Image.fromarray(overlayed[..., ::-1]) #convert BGR to RGB
# --------------------------=============================-------------------------


# -----------------------------------------------------------
# Main Gradio Function 
# ==========================================================================

theme = gr.themes.Ocean(
    secondary_hue="gray",
    neutral_hue="emerald",
    )

#649675 color
css="""
.gradio-container {background-color: #E6ECEF;}


#heading{text-align:center;}
# gradcam_text_1, #gradcam_text_2, #gradcam_text_3{text-align: center;}

#retrieved_results_text_markdown{width:100%;  border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
#retrieved_results_text_markdown{font-size: 100 px; font-weight:ultrabold; text-align: center; color: black;}
#text-boxes{height:250px;}

#query_section{width:100%; background-color: #008098; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }



#result-row-0, #result-row-2, #result-row-4 {
    background-color: #00b3b3;  
    padding: 12px;
    border-radius: 8px;
}

#result-row-1, #result-row-3, #result-row-5 {
    background-color: #008080;  
    padding: 12px;
    border-radius: 8px;
}
"""
#text-boxes{width: 100%; height: 200px; font-size: 14px; color: black; background-color: #f0f0f0; border-radius: 10px; padding: 10px; resize: none;}

# Reference for custom CSS styling:
# gr.Markdown("<div style='color: teal; font-weight: bold;'> Text Here </div>")


# .class_name:nth-child(0dd or even){properties: value;}

with gr.Blocks(title="CBMIR", theme=theme,css=css) as demo:
    # gr.Markdown("##  Content-Based Medical Image Retrieval System",elem_id="heading")
    gr.Markdown("<div style='color:black; font-size:40px; font-weight:bold; text-shadow: 2px 2px 5px rgba(100,255,130,0.5);'>Vision Transformer Based Medical Image Search<br> With Explainable-AI</div>"
                ,elem_id="heading")
    
    gr.Markdown("<p style='color:black; text-align:center; font-size:20px;'>Upload a query image to retrieve similar images with Grad-CAM visualizations.</p>")

    # Row 1: Query Upload + Query Grad-CAM
    with gr.Row(elem_id="query_section"):
        with gr.Column():
            query_input = gr.Image(label=" Upload Query Image", type="pil", width=400, height=400)
            search_button = gr.Button(" 🔍 SEARCH", scale=1)

        with gr.Column():
            query_gradcam_display = gr.Image(label="🔥 Query Image Grad-CAM",width=330,height=350)
            gr.Markdown("<h4 style='color:white; text-align:center;display:block;'>Query Image Visualization</h4>",elem_id="gradcam_text_1")
    # with gr.Row():
    #     # gr.Markdown("search butoon belongs to here")  
    #     search_button = gr.Button("🔍 Search", scale=0.5, elem_id="search_btn")

    # Row 2: Retrieved Results (5 rows x 3 columns)
    retrieved_imgs = []
    retrieved_gradcams = []
    metadata_boxes = []
    # gr.Markdown("<hr>")  # Horizontal line

    gr.Markdown("<div style='height:40px; width:100%; background-color:white;'><p style='color:black; text-align:center; font-size:20px; text-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>TOP FIVE RETRIEVED RESULTS WITH GRAD-CAM VISUALIZATION AND META-DATA</p></div>"
                , elem_id="retrieved_results_text_markdown")
    
    # 5 = number of results to display
    for i in range(6):  # 5 results
        section_id = f"result-row-{i}"  # Unique ID per row
        with gr.Row(elem_id=section_id):
            with gr.Column():
                img = gr.Image(label=f"Retrieved Image {i+1}", width=324, height=300)
                retrieved_imgs.append(img)
            with gr.Column():
                grad = gr.Image(label=f"Grad-CAM {i+1}", width=270, height=280)
                # Add a Markdown label for Grad-CAM
                # gr.Markdown(f"This is the Grad-CAM {i+1}", elem_id="gradcam_text_{i+1}")
                
                retrieved_gradcams.append(grad)
            with gr.Column():
                meta = gr.Textbox(label=f"📄 Metadata of image  {i+1}",lines=8, interactive=False, elem_id="text-boxes")
                metadata_boxes.append(meta)
        gr.Markdown("<div style='background-color:white; height:5px; width:100%;'>   </div>")  # Horizontal line
            

    # Event logic
    def run_cbmir(query_img):
        print("run_cbmir function called by click event")
        if query_img is None:
            raise gr.Error("Please upload a query image!")

        import tempfile, uuid, os
        tmp_path = os.path.join(tempfile.gettempdir(), f"query_{uuid.uuid4().hex}.png")
        query_img.save(tmp_path)
        
        try:
            query_tensor = preprocess_query_image(tmp_path)
            query_feature = extract_query_features(vit_model, query_tensor)
            results = search_faiss_index(query_feature, faiss_index, metadata, top_k=6)
            # top_k=5 is the number of results to return by the search function
            query_gradcam = get_overlayed_gradcam_image(tmp_path, vit_model)

            # Prepare all 5 retrieved results
            retrieved_imgs_out = []
            retrieved_gradcams_out = []
            metadata_out = []

            for res in results:
                img = Image.open(res["image_path"])
                grad_img = get_overlayed_gradcam_image(res["image_path"], vit_model)
                meta = (
                    f"Rank: {res['rank']}\n"
                    f"Name: {res['image_name']}\n"
                    f"Distance: {res['distance']:.4f}\n"
                    f"Similarity: {res['similarity_score']*100:.2f}% \n"
                    f"Image Path:\n {res['image_path']}"
                )
                retrieved_imgs_out.append(img)
                retrieved_gradcams_out.append(grad_img)
                metadata_out.append(meta)

            # Fill up remaining rows if < 5
            # imp No. of retrieved images returned by the search function is required.
            while len(retrieved_imgs_out) < 5:
                retrieved_imgs_out.append(None)
                retrieved_gradcams_out.append(None)
                metadata_out.append("")

            return [query_gradcam] + retrieved_imgs_out + retrieved_gradcams_out + metadata_out

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Set up search button click
    search_button.click(
        fn=run_cbmir,
        inputs=[query_input],
        outputs=[query_gradcam_display] + retrieved_imgs + retrieved_gradcams + metadata_boxes
    )

demo.launch()
