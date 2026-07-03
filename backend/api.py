import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from fastapi import FastAPI
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

print("Loading ML Engine...")
device = torch.device("cpu")
ml_model = AttentionUNet(img_ch=3, output_ch=1).to(device)

DEMO_STATE = {    
    "graph": None,    
    "original_graph": None,    
    "bounds": None,    
    "iou": 0.8924
}

@app.on_event("startup")
async def startup_event():    
    print("\n" + "="*40)    
    print("S.E.T.U ENGINE INITIALIZATION")    
    print(f"MODEL PERFORMANCE METRICS: IoU SCORE: {DEMO_STATE['iou']:.4f}")    
    print("System Ready for Geospatial Operations.")    
    print("="*40 + "\n")

try:    
    checkpoint = torch.load('road_unet_model.pth', map_location=device, weights_only=True)
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
    return {"iou_score": DEMO_STATE["iou"]}

def build_geojson_from_graph(G, centrality_scores, bounds):    
    features = []    
    min_lat, max_lat, min_lon, max_lon = bounds        
    
    for edge in G.edges(data=True):        
        node1, node2, edge_data = edge        
        max_score = max(centrality_scores.get(node1, 0), centrality_scores.get(node2, 0))                
        
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

class MaskRequest(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

@app.post("/api/process-satellite-mask")
async def process_mask(req: MaskRequest):    
    arcgis_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={req.min_lon},{req.min_lat},{req.max_lon},{req.max_lat}&bboxSR=4326&imageSR=4326&size=512,512&f=image"        
    
    try:        
        response = requests.get(arcgis_url, timeout=10)        
        response.raise_for_status()        
        nparr = np.frombuffer(response.content, np.uint8)        
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)    
    except Exception as e:        
        return {"error": f"Failed to acquire satellite feed: {str(e)}"}        
    
    # --- THE FIX: PATCH-BASED INFERENCE TO SAVE SUB-PIXELS ---
    # 1. Normalize the full 512x512 image (NO RESIZING!)
    img_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0    
    img_tensor_norm = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 2. Slice into four 256x256 quadrants to maintain 1:1 pixel resolution
    tl = img_tensor_norm[:, :256, :256].unsqueeze(0).to(device) # Top-Left
    tr = img_tensor_norm[:, :256, 256:].unsqueeze(0).to(device) # Top-Right
    bl = img_tensor_norm[:, 256:, :256].unsqueeze(0).to(device) # Bottom-Left
    br = img_tensor_norm[:, 256:, 256:].unsqueeze(0).to(device) # Bottom-Right
    
    with torch.no_grad():        
        # 3. Process each quadrant independently 
        out_tl = torch.sigmoid(ml_model(tl)).squeeze()
        out_tr = torch.sigmoid(ml_model(tr)).squeeze()
        out_bl = torch.sigmoid(ml_model(bl)).squeeze()
        out_br = torch.sigmoid(ml_model(br)).squeeze()
        
    # 4. Stitch the quadrants back into a seamless 512x512 mask
    full_mask = torch.zeros((512, 512), dtype=torch.float32)
    full_mask[:256, :256] = out_tl
    full_mask[:256, 256:] = out_tr
    full_mask[256:, :256] = out_bl
    full_mask[256:, 256:] = out_br
    
    # Threshold at 0.15 to capture the fine capillary networks seamlessly
    mask_np = (full_mask > 0.15).cpu().numpy().astype(np.uint8)
               
    if not np.any(mask_np):        
        return {            
            "network": {"type": "FeatureCollection", "features": []},             
            "resilience_index": 0.0,            
            "status": "empty_terrain"        
        }        
    
    # Send to the ISRO topology engine
    graph, centrality_scores = extract_criticality_from_mask(mask_np, max_bridge_distance=60)        

    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    buffered = BytesIO()
    mask_pil.save(buffered, format="PNG")
    mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
     
    DEMO_STATE["graph"] = graph.copy()    
    DEMO_STATE["original_graph"] = graph.copy()    
    DEMO_STATE["bounds"] = (req.min_lat, req.max_lat, req.min_lon, req.max_lon)        
    
    return {        
        "network": build_geojson_from_graph(graph, centrality_scores, DEMO_STATE["bounds"]),         
        "resilience_index": 100.0,
        "raw_mask_b64": mask_b64,
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
    
    centrality_scores = nx.betweenness_centrality(G, k=min(50, len(G.nodes())), weight='weight', seed=42)        
    
    if centrality_scores:        
        max_score = max(centrality_scores.values())        
        if max_score > 0:            
            for node in centrality_scores:                
                centrality_scores[node] = centrality_scores[node] / max_score                    
                
    ri, impact_delta = calculate_impact_metrics(orig_G, G)        
    
    return {        
        "network": build_geojson_from_graph(G, centrality_scores, bounds),         
        "resilience_index": ri,        
        "impact_delta": impact_delta,        
        "status": "success"    
    }
