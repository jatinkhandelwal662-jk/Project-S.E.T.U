import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
from pydantic import BaseModel
import requests
import base64
from io import BytesIO
from PIL import Image

from graph_engine import extract_criticality_from_mask, calculate_impact_metrics
from model import AttentionUNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

print("Loading TARS ML Engine...")
device = torch.device("cpu")
ml_model = AttentionUNet(img_ch=3, output_ch=1).to(device)

DEMO_STATE = {    
    "graph": None,    
    "original_graph": None,    
    "bounds": None,    
    "iou": 0.8924 # Stored evaluation metric
}

@app.on_event("startup")
async def startup_event():    
    print("\n" + "="*40)    
    print("S.E.T.U ENGINE INITIALIZATION")    
    print(f"MODEL PERFORMANCE METRICS: IoU SCORE: {DEMO_STATE['iou']:.4f}")    
    print("System Ready for Geospatial Operations.")    
    print("="*40 + "\n")

try:    
    # Load the raw file securely
    checkpoint = torch.load('road_unet_model.pth', map_location=device, weights_only=True)
    
    # Smart-extract the weights whether it's a training checkpoint or a final deployment file    
    if 'model_state_dict' in checkpoint:        
        ml_model.load_state_dict(checkpoint['model_state_dict'])    
    else:        
        ml_model.load_state_dict(checkpoint)    
        
    ml_model.eval()    
    print("Neural Network Loaded Successfully.")
except FileNotFoundError:    
    print("WARNING: road_unet_model.pth not found. Ensure model is in the same directory.")

@app.get("/api/metrics")
async def get_metrics():    
    """Endpoint for the frontend to fetch model evaluation data"""    
    return {"iou_score": DEMO_STATE["iou"]}

def build_geojson_from_graph(G, centrality_scores, bounds):    
    """Dynamically maps the 512x512 AI mask to real-world GPS coordinates."""    
    features = []    
    min_lat, max_lat, min_lon, max_lon = bounds        
    
    for edge in G.edges(data=True):        
        node1, node2, edge_data = edge        
        max_score = max(centrality_scores.get(node1, 0), centrality_scores.get(node2, 0))                
        
        # Convert 512x512 matrix pixels directly to GPS coordinates        
        lon1 = min_lon + (node1[1] / 512.0) * (max_lon - min_lon)        
        lat1 = max_lat - (node1[0] / 512.0) * (max_lat - min_lat)        
        lon2 = min_lon + (node2[1] / 512.0) * (max_lon - min_lon)        
        lat2 = max_lat - (node2[0] / 512.0) * (max_lat - min_lat)                
        
        features.append({            
            "type": "Feature",            
            "properties": {                
                "criticality_score": float(max_score),                
                "pixel_n1": [int(node1[0]), int(node1[1])],                
                "pixel_n2": [int(node2[0]), int(node2[1])]            
            },            
            "geometry": {                
                "type": "LineString",                
                "coordinates": [[lon1, lat1], [lon2, lat2]]             
            }        
        })    
    return {"type": "FeatureCollection", "features": features}

@app.post("/api/process-satellite-mask")
async def process_mask(    
    min_lat: float = Form(...),     
    max_lat: float = Form(...),    
    min_lon: float = Form(...),    
    max_lon: float = Form(...)
):    
    # 1. AUTONOMOUS SATELLITE FEED ACQUISITION    
    arcgis_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326&size=512,512&f=image"        
    
    try:        
        response = requests.get(arcgis_url, timeout=10)        
        response.raise_for_status()        
        nparr = np.frombuffer(response.content, np.uint8)        
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)    
    except Exception as e:        
        return {"error": f"Failed to acquire satellite feed: {str(e)}"}        
    
    # 2. AI PROCESSING (With Normalization & Interpolation Fix)    
    img_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0    
    
    # Resize to 256x256 to match the Transformer training parameters
    img_tensor_256 = TF.resize(img_tensor, [256, 256], antialias=True)        
    
    # Apply the exact normalization filter used during training
    img_tensor_norm = TF.normalize(
        img_tensor_256, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ).unsqueeze(0).to(device)
    
    with torch.no_grad():        
        raw_logits = ml_model(img_tensor_norm)        
        
        # Convert raw math to a 0-1 probability map
        probabilities = torch.sigmoid(raw_logits)
        
        # Resize the 256x256 mask back to the 512x512 tile size
        probabilities_resized = F.interpolate(probabilities, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Threshold to create the binary mask
        mask_np = (probabilities_resized > 0.4).float().cpu().squeeze().numpy().astype(np.uint8)
            
    # 3. EDGE CASE: Blank Image / No Roads    
    if not np.any(mask_np):        
        return {            
            "network": {"type": "FeatureCollection", "features": []},             
            "resilience_index": 0.0,            
            "status": "empty_terrain"        
        }        
    
    # 4. Topology Extraction    
    graph, centrality_scores = extract_criticality_from_mask(mask_np)        

    # --- ISRO PHASE IV: GENERATE BASE64 AI MASK FOR FRONTEND TOGGLE ---
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    buffered = BytesIO()
    mask_pil.save(buffered, format="PNG")
    mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # ------------------------------------------------------------------
    
    # 5. Save State for Disaster Simulation    
    DEMO_STATE["graph"] = graph.copy()    
    DEMO_STATE["original_graph"] = graph.copy()    
    DEMO_STATE["bounds"] = (min_lat, max_lat, min_lon, max_lon)        
    
    return {        
        "network": build_geojson_from_graph(graph, centrality_scores, DEMO_STATE["bounds"]),         
        "resilience_index": 100.0,
        "raw_mask_b64": mask_b64, # Send the vision mask to the dashboard
        "status": "success"    
    }

class EdgeAblationRequest(BaseModel):    
    n1_y: int    
    n1_x: int    
    n2_y: int    
    n2_x: int

@app.post("/api/ablate-edge")
async def ablate_edge(req: EdgeAblationRequest):    
    G = DEMO_STATE.get("graph")    
    orig_G = DEMO_STATE.get("original_graph")    
    bounds = DEMO_STATE.get("bounds")        
    
    if G is None or orig_G is None:        
        return {"error": "No active network loaded."}        
        
    edge = ((req.n1_y, req.n1_x), (req.n2_y, req.n2_x))    
    if G.has_edge(*edge): G.remove_edge(*edge)    
    elif G.has_edge(edge[1], edge[0]): G.remove_edge(edge[1], edge[0])        
    
    # Recalculate Centrality after bridge collapse    
    centrality_scores = nx.betweenness_centrality(G, k=min(50, len(G.nodes())), weight='weight')        
    
    # Normalize scores to keep colors bright    
    if centrality_scores:        
        max_score = max(centrality_scores.values())        
        if max_score > 0:            
            for node in centrality_scores:                
                centrality_scores[node] = centrality_scores[node] / max_score                    
                
    # Fetch both Resilience Index and Impact Delta    
    ri, impact_delta = calculate_impact_metrics(orig_G, G)        
    
    return {        
        "network": build_geojson_from_graph(G, centrality_scores, bounds),         
        "resilience_index": ri,        
        "impact_delta": impact_delta,        
        "status": "success"    
    }
    
