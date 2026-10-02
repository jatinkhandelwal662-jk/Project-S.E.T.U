import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import requests
import base64
from io import BytesIO
from PIL import Image

# Consolidating all imports from graph_engine
from graph_engine import (
    extract_criticality_from_mask, 
    calculate_impact_metrics, 
    compute_emergency_route, 
    haversine_distance, 
    pixel_to_latlon
)
from model import AttentionUNet

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = torch.device("cpu")
ml_model = AttentionUNet(img_ch=3, output_ch=1).to(device)

DEMO_STATE = { 
    "graph": None, 
    "original_graph": None, 
    "bounds": None, 
    "iou": 0.8924, 
    "stats": None,
    "ambulance_start": None,
    "ambulance_end": None
}

try:    
    checkpoint = torch.load('road_unet_model.pth', map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint: 
        ml_model.load_state_dict(checkpoint['model_state_dict'])    
    else: 
        ml_model.load_state_dict(checkpoint)    
    ml_model.eval()    
except FileNotFoundError: 
    print("Warning: road_unet_model.pth not found. Running without weights.")

def encode_image(img_arr):
    pil_img = Image.fromarray(img_arr)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
                "edge_type": edge_data.get('type', 'actual'),
                "node_id": f"N-{abs(hash(node1)) % 9999}",
                "impact_multiplier": float(max_score * 1.5),
                "pixel_n1": [int(node1[0]), int(node1[1])],                
                "pixel_n2": [int(node2[0]), int(node2[1])]            
            },            
            "geometry": { "type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]] }        
        })    
    return {"type": "FeatureCollection", "features": features}

@app.post("/api/process-satellite-mask")
async def process_mask(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON received by server."}
        
    min_lat = float(data.get("min_lat", 28.61))
    max_lat = float(data.get("max_lat", 28.62))
    min_lon = float(data.get("min_lon", 77.02))
    max_lon = float(data.get("max_lon", 77.03))

    arcgis_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=4326&size=512,512&f=image"        
    
    try:        
        response = requests.get(arcgis_url, timeout=10)        
        response.raise_for_status()        
        nparr = np.frombuffer(response.content, np.uint8)        
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)        
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)    
    except Exception as e: 
        return {"error": f"Failed to acquire satellite feed: {str(e)}"}        
    
    img_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0    
    img_tensor_norm = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    tl = img_tensor_norm[:, :256, :256].unsqueeze(0).to(device)
    tr = img_tensor_norm[:, :256, 256:].unsqueeze(0).to(device)
    bl = img_tensor_norm[:, 256:, :256].unsqueeze(0).to(device)
    br = img_tensor_norm[:, 256:, 256:].unsqueeze(0).to(device)
    
    with torch.no_grad():        
        out_tl = torch.sigmoid(ml_model(tl)).squeeze()
        out_tr = torch.sigmoid(ml_model(tr)).squeeze()
        out_bl = torch.sigmoid(ml_model(bl)).squeeze()
        out_br = torch.sigmoid(ml_model(br)).squeeze()
        
    full_mask = torch.zeros((512, 512), dtype=torch.float32)
    full_mask[:256, :256] = out_tl
    full_mask[:256, 256:] = out_tr
    full_mask[256:, :256] = out_bl
    full_mask[256:, 256:] = out_br
    
    mask_np = (full_mask > 0.15).cpu().numpy().astype(np.uint8)
               
    if not np.any(mask_np): 
        return {"network": {"type": "FeatureCollection", "features": []}, "status": "empty_terrain"}        
    
    graph, centrality_scores, topo_stats, skeleton = extract_criticality_from_mask(mask_np, max_bridge_distance=60)        
    
    DEMO_STATE["graph"] = graph.copy()    
    DEMO_STATE["original_graph"] = graph.copy()    
    DEMO_STATE["bounds"] = (min_lat, max_lat, min_lon, max_lon) 
    DEMO_STATE["stats"] = topo_stats       
    DEMO_STATE["ambulance_start"] = None
    DEMO_STATE["ambulance_end"] = None

    return {        
        "network": build_geojson_from_graph(graph, centrality_scores, DEMO_STATE["bounds"]),         
        "status": "success"    
    }

@app.post("/api/emergency-route")
async def emergency_route(request: Request):
    G = DEMO_STATE.get("graph")
    bounds = DEMO_STATE.get("bounds")
    
    if G is None or bounds is None:
        return {"error": "No active network loaded."}
        
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON payload."}
        
    start_lat = float(data.get("start_lat"))
    start_lon = float(data.get("start_lon"))
    end_lat = float(data.get("end_lat"))
    end_lon = float(data.get("end_lon"))

    DEMO_STATE["ambulance_start"] = (start_lat, start_lon)
    DEMO_STATE["ambulance_end"] = (end_lat, end_lon)

    route_geojson, dist_km, eta_mins, msg = compute_emergency_route(
        G, (start_lat, start_lon), (end_lat, end_lon), bounds
    )
    
    return {
        "route": route_geojson,
        "distance_km": dist_km,
        "eta_mins": eta_mins,
        "status_message": msg,
        "status": "success" if route_geojson else "severed"
    }

@app.post("/api/simulate-collapse")
async def simulate_collapse(request: Request):
    G = DEMO_STATE.get("graph")
    bounds = DEMO_STATE.get("bounds")
    
    if G is None or bounds is None:
        return {"error": "No active network loaded."}
        
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON payload."}
        
    disaster_lat = float(data.get("lat"))
    disaster_lon = float(data.get("lon"))
    # Adjusted default to 50 meters for a precise bridge/road collapse
    radius_km = float(data.get("radius_km", 0.05)) 

    nodes_to_remove = []
    
    for node in list(G.nodes()):
        n_lat, n_lon = pixel_to_latlon(node, bounds)
        dist = haversine_distance(disaster_lat, disaster_lon, n_lat, n_lon)
        if dist <= radius_km:
            nodes_to_remove.append(node)
            
    G.remove_nodes_from(nodes_to_remove)
    
    rerouted_ambulance = None
    if DEMO_STATE.get("ambulance_start") and DEMO_STATE.get("ambulance_end"):
        route_geojson, dist_km, eta_mins, msg = compute_emergency_route(
            G, DEMO_STATE["ambulance_start"], DEMO_STATE["ambulance_end"], bounds
        )
        rerouted_ambulance = {
            "route": route_geojson,
            "distance_km": dist_km,
            "eta_mins": eta_mins,
            "status_message": "REROUTED AROUND COLLAPSE" if route_geojson else "CRITICAL: NO SURVIVING ROUTES!"
        }

    centrality_scores = nx.betweenness_centrality(G, k=min(20, len(G.nodes())), weight='weight', seed=42)
    
    return {
        "network": build_geojson_from_graph(G, centrality_scores, bounds),
        "rerouted_ambulance": rerouted_ambulance,
        "nodes_destroyed": len(nodes_to_remove),
        "status": "success"
    }