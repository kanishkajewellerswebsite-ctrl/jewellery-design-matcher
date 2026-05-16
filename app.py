"""
AI Design Matcher - Complete Cloud Backup Version
Images in Cloudinary, Metadata in Supabase
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
import time
from datetime import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import BytesIO

# Computer Vision
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

# Cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Supabase
from supabase import create_client, Client

# ============================================
# 1. PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Design Matcher - Kanishka Jewellers",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 2. CONSTANTS
# ============================================
COMPANY_NAME = "Kanishka Jewellers Pvt Ltd"
COMPANY_PHONE = "+91 9829194093, +91 9680434748"
COMPANY_EMAIL = "kanishkajewellers@gmail.com"
COMPANY_WEBSITE = "www.kanishkajewellers.com"

CATEGORY_COLORS = {
    "Ring": "#FF6B6B",
    "Necklace": "#4ECDC4",
    "Earring": "#45B7D1",
    "Bracelet": "#96CEB4",
    "Pendant": "#FFE66D",
    "Brooch": "#FF9F1C",
    "Cufflink": "#C77DFF",
    "Other": "#A8A8A8"
}

CATEGORIES = ['Ring', 'Necklace', 'Earring', 'Bracelet', 'Pendant', 'Brooch', 'Cufflink', 'Other']
METAL_TYPES = ['Gold', 'Silver', 'Platinum', 'Rose Gold', 'White Gold']
STONE_TYPES = ['Diamond', 'Emerald', 'Ruby', 'Sapphire', 'Pearl', 'Opal', 'Jade', 'None', 'Other']

# ============================================
# 3. SESSION STATE
# ============================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'gallery_page' not in st.session_state:
    st.session_state.gallery_page = 1
if 'selected_design' not in st.session_state:
    st.session_state.selected_design = None
if 'expert_detector' not in st.session_state:
    st.session_state.expert_detector = None
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False
if 'design_embeddings' not in st.session_state:
    st.session_state.design_embeddings = {}

# ============================================
# 4. SUPABASE INITIALIZATION
# ============================================

def init_supabase():
    """Initialize Supabase connection"""
    try:
        if 'SUPABASE_URL' in st.secrets and 'SUPABASE_KEY' in st.secrets:
            supabase = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_KEY"]
            )
            return supabase
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ Supabase not configured: {e}")
        return None

supabase = init_supabase()

# ============================================
# 5. SUPABASE DATABASE FUNCTIONS
# ============================================

def get_all_designs_supabase():
    """Get all designs from Supabase"""
    if supabase is None:
        return []
    try:
        response = supabase.table("designs").select("*").execute()
        return response.data
    except Exception as e:
        print(f"Error fetching designs: {e}")
        return []

def get_design_by_no_supabase(design_no):
    """Get single design by number from Supabase"""
    if supabase is None:
        return None
    try:
        response = supabase.table("designs").select("*").eq("design_no", design_no).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error fetching design: {e}")
        return None

def save_design_to_supabase(design_no, design_name, category, metal_type, stone_type, images, description="", price_range=""):
    """Save design to Supabase cloud database"""
    if supabase is None:
        return False
    
    try:
        images_json = json.dumps(images)
        
        data = {
            "design_no": design_no,
            "design_name": design_name,
            "category": category,
            "metal_type": metal_type,
            "stone_type": stone_type,
            "images": images_json,
            "description": description,
            "price_range": price_range,
            "date_added": datetime.now().isoformat(),
            "status": "Active",
            "view_count": 0,
            "storage_type": "cloudinary"
        }
        
        response = supabase.table("designs").upsert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving to Supabase: {e}")
        return False

def update_design_in_supabase(design_no, design_name, category, metal_type, stone_type, description="", price_range=""):
    """Update design in Supabase"""
    if supabase is None:
        return False
    
    try:
        data = {
            "design_name": design_name,
            "category": category,
            "metal_type": metal_type,
            "stone_type": stone_type,
            "description": description,
            "price_range": price_range
        }
        response = supabase.table("designs").update(data).eq("design_no", design_no).execute()
        return True
    except Exception as e:
        print(f"Error updating design: {e}")
        return False

def delete_design_from_supabase(design_no):
    """Delete design from Supabase"""
    if supabase is None:
        return False
    
    try:
        response = supabase.table("designs").delete().eq("design_no", design_no).execute()
        return True
    except Exception as e:
        print(f"Error deleting from Supabase: {e}")
        return False

def increment_view_count_supabase(design_no):
    """Increment view count in Supabase"""
    if supabase is None:
        return
    
    try:
        # Get current count
        design = get_design_by_no_supabase(design_no)
        if design:
            current_count = design.get('view_count', 0)
            supabase.table("designs").update({"view_count": current_count + 1}).eq("design_no", design_no).execute()
    except Exception as e:
        print(f"Error updating view count: {e}")

def get_all_categories_supabase():
    """Get unique categories from Supabase"""
    designs = get_all_designs_supabase()
    categories = list(set([d.get('category') for d in designs if d.get('category')]))
    return sorted(categories) if categories else CATEGORIES

def get_all_metal_types_supabase():
    """Get unique metal types from Supabase"""
    designs = get_all_designs_supabase()
    metals = list(set([d.get('metal_type') for d in designs if d.get('metal_type')]))
    return sorted(metals) if metals else METAL_TYPES

def get_all_stone_types_supabase():
    """Get unique stone types from Supabase"""
    designs = get_all_designs_supabase()
    stones = list(set([d.get('stone_type') for d in designs if d.get('stone_type')]))
    return sorted(stones) if stones else STONE_TYPES

def get_total_count_supabase(category=None, metal=None, stone=None, search_term=None):
    """Get total count with filters from Supabase"""
    designs = get_all_designs_supabase()
    
    filtered = [d for d in designs if d.get('status') == 'Active']
    
    if category and category != "All":
        filtered = [d for d in filtered if d.get('category') == category]
    if metal and metal != "All":
        filtered = [d for d in filtered if d.get('metal_type') == metal]
    if stone and stone != "All":
        filtered = [d for d in filtered if d.get('stone_type') == stone]
    if search_term:
        search_term = search_term.lower()
        filtered = [d for d in filtered if search_term in d.get('design_no', '').lower() or search_term in d.get('design_name', '').lower()]
    
    return len(filtered)

def get_designs_paginated_supabase(page=1, page_size=12, category=None, metal=None, stone=None, search_term=None):
    """Get paginated designs from Supabase"""
    designs = get_all_designs_supabase()
    
    filtered = [d for d in designs if d.get('status') == 'Active']
    
    if category and category != "All":
        filtered = [d for d in filtered if d.get('category') == category]
    if metal and metal != "All":
        filtered = [d for d in filtered if d.get('metal_type') == metal]
    if stone and stone != "All":
        filtered = [d for d in filtered if d.get('stone_type') == stone]
    if search_term:
        search_term = search_term.lower()
        filtered = [d for d in filtered if search_term in d.get('design_no', '').lower() or search_term in d.get('design_name', '').lower()]
    
    # Sort by date_added (newest first)
    filtered.sort(key=lambda x: x.get('date_added', ''), reverse=True)
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    
    return filtered[start:end]

# ============================================
# 6. RGBA CONVERSION FUNCTION
# ============================================

def convert_to_rgb(image):
    """Convert image to RGB mode (handles RGBA, P, LA modes)"""
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        return rgb_image
    elif image.mode in ('P', 'LA'):
        return image.convert('RGB')
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image

# ============================================
# 7. CLOUDINARY FUNCTIONS
# ============================================

def init_cloudinary():
    """Initialize Cloudinary"""
    try:
        if 'CLOUDINARY_CLOUD_NAME' in st.secrets:
            cloudinary.config(
                cloud_name=st.secrets["CLOUDINARY_CLOUD_NAME"],
                api_key=st.secrets["CLOUDINARY_API_KEY"],
                api_secret=st.secrets["CLOUDINARY_API_SECRET"],
                secure=True
            )
            return True
    except:
        pass
    return False

def upload_to_cloudinary(image_file, design_no, angle=1):
    """Upload image to Cloudinary"""
    try:
        upload_result = cloudinary.uploader.upload(
            image_file,
            folder=f"jewellery_designs/{design_no}",
            public_id=f"angle_{angle}",
            overwrite=True,
            tags=[design_no, "jewellery"],
            transformation=[
                {'width': 800, 'height': 800, 'crop': 'limit'},
                {'quality': 'auto'},
                {'fetch_format': 'auto'}
            ]
        )
        return upload_result['secure_url']
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def upload_multiple_to_cloudinary(images, design_no):
    """Upload multiple images to Cloudinary"""
    urls = []
    for i, img in enumerate(images[:4]):
        img = convert_to_rgb(img)
        temp_path = f"temp_{design_no}_{i}.jpg"
        img.save(temp_path, "JPEG", quality=85)
        
        with open(temp_path, 'rb') as f:
            url = upload_to_cloudinary(f, design_no, i+1)
            if url:
                urls.append(url)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return urls

def delete_from_cloudinary(design_no):
    """Delete images from Cloudinary"""
    try:
        cloudinary.api.delete_resources_by_prefix(f"jewellery_designs/{design_no}")
        cloudinary.api.delete_folder(f"jewellery_designs/{design_no}")
        return True
    except:
        return False

def is_cloudinary_url(url):
    return url and url.startswith('https://res.cloudinary.com')

# ============================================
# 8. HELPER FUNCTIONS
# ============================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_password():
    password_file = "admin_password.json"
    if not os.path.exists(password_file):
        default_hash = hash_password("admin123")
        with open(password_file, "w") as f:
            json.dump({"password": default_hash}, f)
    with open(password_file, "r") as f:
        return json.load(f)["password"]

def verify_password(password):
    stored_hash = init_password()
    return hash_password(password) == stored_hash

# ============================================
# 9. EXPERT DETECTOR
# ============================================

@st.cache_resource
def load_expert_detector():
    try:
        from expert_detector import ExpertJewelryDetector
        return ExpertJewelryDetector(use_gpu=False)
    except:
        return None

def load_precomputed_embeddings():
    embedding_file = "embeddings/expert_embeddings.pkl"
    if os.path.exists(embedding_file):
        try:
            import pickle
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f)
            st.session_state.design_embeddings = embeddings
            st.session_state.embeddings_loaded = True
        except:
            pass

# ============================================
# 10. AI MATCHING FUNCTIONS
# ============================================

def extract_features_from_image(image):
    gray = np.array(image.convert('L'))
    gray = cv2.resize(gray, (256, 256))
    
    features = []
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = hist.flatten() / hist.sum()
    features.extend(hist)
    
    h, w = gray.shape
    for i in range(4):
        for j in range(4):
            cell = gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            features.append(np.mean(cell) / 255.0)
            features.append(np.std(cell) / 255.0)
    
    return np.array(features)

def find_similar_designs_expert(query_image, detector, top_k=8):
    results = []
    all_designs = get_all_designs_supabase()
    
    if detector is not None and st.session_state.embeddings_loaded:
        query_emb = detector.generate_embedding(query_image)
        for design in all_designs:
            design_no = design.get('design_no')
            if design_no and design_no in st.session_state.design_embeddings:
                design_emb = st.session_state.design_embeddings[design_no]
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([query_emb], [design_emb])[0][0]
                results.append((design, similarity))
    else:
        query_features = extract_features_from_image(query_image)
        for design in all_designs:
            images = json.loads(design.get('images', '[]'))
            if images:
                try:
                    img_url = images[0]
                    if is_cloudinary_url(img_url):
                        response = requests.get(img_url)
                        img = Image.open(BytesIO(response.content))
                    else:
                        img = Image.open(img_url)
                    
                    if img:
                        design_features = extract_features_from_image(img)
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarity = cosine_similarity([query_features], [design_features])[0][0]
                        results.append((design, similarity))
                except:
                    pass
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# ============================================
# 11. UI COMPONENTS
# ============================================

def display_design_card(design, similarity=None):
    category = design.get('category', 'Other')
    category_color = CATEGORY_COLORS.get(category, "#A8A8A8")
    
    images = json.loads(design.get('images', '[]'))
    
    if images:
        image_url = images[0]
        if is_cloudinary_url(image_url):
            st.image(image_url, use_column_width=True)
        elif os.path.exists(image_url):
            st.image(Image.open(image_url), use_column_width=True)
        else:
            st.image("https://placehold.co/400x300/1A2634/D4AF37?text=No+Image", use_column_width=True)
    else:
        st.image("https://placehold.co/400x300/1A2634/D4AF37?text=No+Image", use_column_width=True)
    
    st.markdown(f"""
    <div style="text-align: center;">
        <h4>💎 {design.get('design_no', 'Unknown')}</h4>
        <p><strong>{design.get('design_name', 'Unknown')}</strong></p>
        <p><span style="background: {category_color}; padding: 2px 8px; border-radius: 12px;">{category}</span></p>
        <p style="font-size: 0.8rem;">Metal: {design.get('metal_type', 'N/A')}<br>Stone: {design.get('stone_type', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if similarity is not None:
        similarity_pct = similarity * 100
        if similarity_pct > 80:
            st.markdown(f"<p style='text-align: center; color: #4CAF50; font-weight: bold;'>{similarity_pct:.1f}% Match 🔥</p>", unsafe_allow_html=True)
        elif similarity_pct > 60:
            st.markdown(f"<p style='text-align: center; color: #FF9800; font-weight: bold;'>{similarity_pct:.1f}% Match ⭐</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; color: #9E9E9E; font-weight: bold;'>{similarity_pct:.1f}% Match 💎</p>", unsafe_allow_html=True)
    
    if st.button(f"🔍 View Details", key=f"view_{design.get('design_no')}"):
        st.session_state.selected_design = design.get('design_no')
        increment_view_count_supabase(design.get('design_no'))
        st.rerun()

def display_design_details(design_no):
    design = get_design_by_no_supabase(design_no)
    if not design:
        return
    
    st.markdown("---")
    st.markdown(f"## 📋 Design Details: {design_no}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        images = json.loads(design.get('images', '[]'))
        if images:
            image_url = images[0]
            if is_cloudinary_url(image_url):
                st.image(image_url, use_column_width=True)
            elif os.path.exists(image_url):
                st.image(Image.open(image_url), use_column_width=True)
    
    with col2:
        st.markdown(f"""
        **Design Name:** {design.get('design_name', 'N/A')}  
        **Category:** {design.get('category', 'N/A')}  
        **Metal Type:** {design.get('metal_type', 'N/A')}  
        **Stone Type:** {design.get('stone_type', 'N/A')}  
        **Description:** {design.get('description', 'N/A')}  
        **Price Range:** {design.get('price_range', 'N/A')}  
        **Date Added:** {design.get('date_added', 'N/A')}  
        **Views:** {design.get('view_count', 0)}  
        """)
    
    if st.button("← Back to Gallery"):
        st.session_state.selected_design = None
        st.rerun()

def display_contact():
    with st.expander("📞 Contact Information", expanded=False):
        st.markdown(f"""
        **Kanishka Jewellers Pvt Ltd**  
        📞 {COMPANY_PHONE}  
        ✉️ {COMPANY_EMAIL}  
        🌐 {COMPANY_WEBSITE}
        """)

# ============================================
# 12. ADMIN FUNCTIONS
# ============================================

def admin_add_design():
    st.markdown("### Add New Design")
    st.info("📸 You can upload up to 4 images per design")
    
    col1, col2 = st.columns(2)
    with col1:
        new_design_no = st.text_input("Design Number *", placeholder="e.g., R001, N002", key="add_no")
        new_design_name = st.text_input("Design Name *", placeholder="e.g., Solitaire Ring", key="add_name")
        new_category = st.selectbox("Category", CATEGORIES, key="add_cat")
    with col2:
        new_metal = st.selectbox("Metal Type", METAL_TYPES, key="add_metal")
        new_stone = st.selectbox("Stone Type", STONE_TYPES, key="add_stone")
    
    new_description = st.text_area("Description", placeholder="Describe the design...", key="add_desc")
    new_price = st.text_input("Price Range", placeholder="e.g., ₹50,000 - ₹1,00,000", key="add_price")
    
    new_images = st.file_uploader(
        "Upload Images (Max 4)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="add_images"
    )
    
    if new_images and len(new_images) > 4:
        st.warning(f"⚠️ You selected {len(new_images)} images. Only the first 4 will be used.")
    
    if new_images:
        preview_cols = st.columns(min(len(new_images), 4))
        for idx, img_file in enumerate(new_images[:4]):
            with preview_cols[idx]:
                img = Image.open(img_file)
                st.image(img, caption=f"Angle {idx+1}")
    
    if st.button("💾 Save Design", type="primary", use_container_width=True, key="add_save"):
        if new_design_no and new_design_name and new_images:
            existing = get_design_by_no_supabase(new_design_no)
            if not existing:
                # Upload to Cloudinary
                if init_cloudinary():
                    images_pil = [Image.open(img) for img in new_images[:4]]
                    image_urls = upload_multiple_to_cloudinary(images_pil, new_design_no)
                    
                    if image_urls:
                        # Save to Supabase
                        if save_design_to_supabase(new_design_no, new_design_name, new_category, 
                                                   new_metal, new_stone, image_urls, 
                                                   new_description, new_price):
                            st.success(f"✅ Design {new_design_no} saved to cloud!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to save to database")
                    else:
                        st.error("❌ Failed to upload images")
                else:
                    st.error("❌ Cloudinary not configured")
            else:
                st.error(f"❌ Design {new_design_no} already exists!")
        else:
            st.error("❌ Please fill all required fields and upload at least one image")

def admin_edit_design():
    st.markdown("### Edit Design")
    
    all_designs = get_all_designs_supabase()
    if all_designs:
        design_options = {f"{d['design_no']} - {d['design_name']}": d['design_no'] for d in all_designs}
        selected = st.selectbox("Select Design to Edit", list(design_options.keys()), key="edit_select")
        design_to_edit = design_options[selected]
        
        if design_to_edit:
            design = get_design_by_no_supabase(design_to_edit)
            if design:
                images = json.loads(design.get('images', '[]'))
                if images:
                    if is_cloudinary_url(images[0]):
                        st.image(images[0], width=200, caption="Current Design")
                
                col1, col2 = st.columns(2)
                with col1:
                    edit_name = st.text_input("Design Name", value=design.get('design_name', ''), key="edit_name")
                    edit_category = st.selectbox("Category", CATEGORIES, 
                                                index=CATEGORIES.index(design.get('category', 'Ring')) if design.get('category') in CATEGORIES else 0,
                                                key="edit_cat")
                with col2:
                    edit_metal = st.selectbox("Metal Type", METAL_TYPES,
                                             index=METAL_TYPES.index(design.get('metal_type', 'Gold')) if design.get('metal_type') in METAL_TYPES else 0,
                                             key="edit_metal")
                    edit_stone = st.selectbox("Stone Type", STONE_TYPES,
                                             index=STONE_TYPES.index(design.get('stone_type', 'None')) if design.get('stone_type') in STONE_TYPES else 0,
                                             key="edit_stone")
                
                edit_description = st.text_area("Description", value=design.get('description', ''), key="edit_desc")
                edit_price = st.text_input("Price Range", value=design.get('price_range', ''), key="edit_price")
                
                if st.button("💾 Update Design", use_container_width=True, key="edit_update"):
                    if update_design_in_supabase(design_to_edit, edit_name, edit_category, edit_metal, edit_stone, edit_description, edit_price):
                        st.success("✅ Design updated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to update design")
    else:
        st.info("No designs to edit")

def admin_delete_design():
    st.markdown("### Delete Design")
    
    all_designs = get_all_designs_supabase()
    if all_designs:
        design_options = {f"{d['design_no']} - {d['design_name']}": d['design_no'] for d in all_designs}
        selected = st.selectbox("Select Design to Delete", list(design_options.keys()), key="delete_select")
        design_to_delete = design_options[selected]
        
        if design_to_delete:
            design = get_design_by_no_supabase(design_to_delete)
            if design:
                images = json.loads(design.get('images', '[]'))
                if images:
                    if is_cloudinary_url(images[0]):
                        st.image(images[0], width=200)
                
                st.warning(f"⚠️ You are about to delete: {design_to_delete} - {design.get('design_name')}")
                
                confirm = st.checkbox("I confirm this deletion", key="delete_confirm")
                if confirm and st.button("🗑️ Permanently Delete", type="primary", use_container_width=True, key="delete_btn"):
                    if delete_design_from_supabase(design_to_delete):
                        delete_from_cloudinary(design_to_delete)
                        st.success("✅ Design deleted successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete design")
    else:
        st.info("No designs to delete")

def admin_settings():
    st.markdown("### System Settings")
    
    st.markdown("#### ☁️ Cloud Storage Status")
    if init_cloudinary():
        st.success("✅ Cloudinary: Connected")
        st.caption(f"Cloud Name: {st.secrets.get('CLOUDINARY_CLOUD_NAME', 'N/A')}")
    else:
        st.error("❌ Cloudinary: Not configured")
    
    if supabase:
        st.success("✅ Supabase: Connected")
    else:
        st.error("❌ Supabase: Not configured")
    
    st.markdown("---")
    st.markdown("#### 🔑 Change Admin Password")
    current_pwd = st.text_input("Current Password", type="password", key="settings_current")
    new_pwd = st.text_input("New Password", type="password", key="settings_new")
    confirm_pwd = st.text_input("Confirm New Password", type="password", key="settings_confirm")
    
    if st.button("Change Password", use_container_width=True, key="settings_change"):
        if verify_password(current_pwd):
            if new_pwd == confirm_pwd and new_pwd:
                hashed = hash_password(new_pwd)
                with open("admin_password.json", "w") as f:
                    json.dump({"password": hashed}, f)
                st.success("✅ Password changed successfully!")
            else:
                st.error("❌ New passwords do not match")
        else:
            st.error("❌ Current password is incorrect")
    
    st.markdown("---")
    st.markdown("#### 📊 Database Statistics")
    
    all_designs = get_all_designs_supabase()
    st.info(f"Total Designs in Cloud: {len(all_designs)}")

# ============================================
# 13. MAIN APP
# ============================================

def main():
    # Load expert detector
    detector = load_expert_detector()
    
    if detector is not None and not st.session_state.embeddings_loaded:
        load_precomputed_embeddings()
    
    # Get categories from database
    db_categories = get_all_categories_supabase()
    db_metals = get_all_metal_types_supabase()
    db_stones = get_all_stone_types_supabase()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #D4AF37; font-size: 2.5rem;">💎 KANISHKA JEWELLERS</h1>
        <h2 style="color: #FFFFFF; font-size: 1.5rem;">AI-Powered Design Studio</h2>
        <hr style="border: 1px solid #D4AF37; opacity: 0.3;">
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔐 Navigation")
        mode = st.radio("Select Mode", ["🔍 Visual Search", "📁 Design Gallery", "⚙️ Admin Panel", "📊 Analytics"], index=0)
        st.markdown("---")
        
        total_designs = len(get_all_designs_supabase())
        st.metric("Total Designs", f"{total_designs:,}")
        st.markdown("---")
        
        if init_cloudinary():
            st.success("✅ Cloud Storage Active")
        else:
            st.warning("⚠️ Local Only")
        
        st.markdown("---")
        
        if detector is not None and st.session_state.embeddings_loaded:
            st.success("✅ AI Expert Active")
        else:
            st.info("ℹ️ Basic Matching")
        
        st.markdown("---")
        display_contact()
    
    # Show design details if selected
    if st.session_state.selected_design:
        display_design_details(st.session_state.selected_design)
        return
    
    # MODE 1: VISUAL SEARCH
    if mode == "🔍 Visual Search":
        st.markdown("## 🔍 Find Similar Designs")
        
        uploaded_file = st.file_uploader("Upload Design Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(query_image, caption="Your Design", use_column_width=True)
            
            with col2:
                if st.button("🔍 Find Similar Designs", type="primary", use_container_width=True):
                    with st.spinner("AI is analyzing your design..."):
                        results = find_similar_designs_expert(query_image, detector, top_k=8)
                        st.session_state.search_results = results
                    
                    if results:
                        st.success(f"✅ Found {len(results)} similar designs!")
                    else:
                        st.warning("⚠️ No similar designs found")
        
        if st.session_state.search_results:
            st.markdown("---")
            st.markdown("## 🎯 Similar Designs Found")
            
            cols = st.columns(4)
            for idx, (design, similarity) in enumerate(st.session_state.search_results):
                with cols[idx % 4]:
                    display_design_card(design, similarity)
            
            if st.button("Clear Results"):
                st.session_state.search_results = []
                st.rerun()
    
    # MODE 2: DESIGN GALLERY
    elif mode == "📁 Design Gallery":
        st.markdown("## 📁 Complete Design Gallery")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            category_filter = st.selectbox("Filter by Category", ["All"] + db_categories, key="gallery_cat")
        with col2:
            metal_filter = st.selectbox("Filter by Metal", ["All"] + db_metals, key="gallery_metal")
        with col3:
            stone_filter = st.selectbox("Filter by Stone", ["All"] + db_stones, key="gallery_stone")
        
        search_term = st.text_input("🔎 Search by Design Number or Name", key="gallery_search")
        
        items_per_page = 12
        total_designs = get_total_count_supabase(category_filter, metal_filter, stone_filter, search_term if search_term else None)
        total_pages = max(1, (total_designs + items_per_page - 1) // items_per_page)
        
        if st.session_state.gallery_page > total_pages:
            st.session_state.gallery_page = total_pages
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        with col1:
            if st.button("⏮ First", disabled=st.session_state.gallery_page <= 1, key="gallery_first"):
                st.session_state.gallery_page = 1
                st.rerun()
        with col2:
            if st.button("◀ Previous", disabled=st.session_state.gallery_page <= 1, key="gallery_prev"):
                st.session_state.gallery_page -= 1
                st.rerun()
        with col3:
            st.markdown(f"<div style='text-align: center;'>Page {st.session_state.gallery_page} of {total_pages}</div>", unsafe_allow_html=True)
        with col4:
            if st.button("Next ▶", disabled=st.session_state.gallery_page >= total_pages, key="gallery_next"):
                st.session_state.gallery_page += 1
                st.rerun()
        with col5:
            if st.button("⏭ Last", disabled=st.session_state.gallery_page >= total_pages, key="gallery_last"):
                st.session_state.gallery_page = total_pages
                st.rerun()
        
        designs = get_designs_paginated_supabase(
            st.session_state.gallery_page, items_per_page,
            category_filter if category_filter != "All" else None,
            metal_filter if metal_filter != "All" else None,
            stone_filter if stone_filter != "All" else None,
            search_term if search_term else None
        )
        
        st.caption(f"Showing {len(designs)} of {total_designs:,} designs")
        
        if designs:
            cols = st.columns(3)
            for idx, design in enumerate(designs):
                with cols[idx % 3]:
                    display_design_card(design)
        else:
            st.info("No designs match the selected filters")
    
    # MODE 3: ADMIN PANEL
    elif mode == "⚙️ Admin Panel":
        st.markdown("## ⚙️ Admin Panel")
        
        if not st.session_state.authenticated:
            st.markdown("### 🔐 Admin Login")
            password = st.text_input("Enter Admin Password", type="password", key="admin_password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True, key="admin_login"):
                    if verify_password(password):
                        st.session_state.authenticated = True
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password")
            with col2:
                if st.button("Demo Mode", use_container_width=True, key="admin_demo"):
                    st.session_state.authenticated = True
                    st.info("🔓 Demo mode - limited access")
                    st.rerun()
        else:
            st.success(f"✅ Logged in as Admin | Total Designs: {len(get_all_designs_supabase()):,}")
            
            if st.button("Logout", use_container_width=True, key="admin_logout"):
                st.session_state.authenticated = False
                st.rerun()
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Design", "✏️ Edit Design", "🗑️ Delete Design", "⚙️ Settings"])
            
            with tab1:
                admin_add_design()
            with tab2:
                admin_edit_design()
            with tab3:
                admin_delete_design()
            with tab4:
                admin_settings()
    
    # MODE 4: ANALYTICS
    elif mode == "📊 Analytics":
        st.markdown("## 📊 Design Analytics Dashboard")
        
        all_designs = get_all_designs_supabase()
        
        if all_designs:
            df_data = []
            for design in all_designs:
                df_data.append({
                    "Design No": design.get('design_no'),
                    "Design Name": design.get('design_name'),
                    "Category": design.get('category'),
                    "Metal Type": design.get('metal_type'),
                    "Stone Type": design.get('stone_type'),
                    "Views": design.get('view_count', 0)
                })
            
            df = pd.DataFrame(df_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Designs", f"{len(df):,}")
            with col2:
                st.metric("Categories", df['Category'].nunique())
            with col3:
                st.metric("Total Views", f"{df['Views'].sum():,}")
            
            st.markdown("---")
            
            cat_counts = df['Category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig = px.bar(cat_counts, x='Category', y='Count', title='Designs by Category',
                        text='Count', color='Count', color_continuous_scale=['#D4AF37', '#9D7EBD'])
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_column_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                metal_counts = df['Metal Type'].value_counts().reset_index()
                metal_counts.columns = ['Metal', 'Count']
                fig = px.pie(metal_counts, values='Count', names='Metal', title='Designs by Metal')
                st.plotly_chart(fig, use_column_width=True)
            
            with col2:
                stone_counts = df['Stone Type'].value_counts().reset_index()
                stone_counts.columns = ['Stone', 'Count']
                fig = px.bar(stone_counts, x='Stone', y='Count', title='Designs by Stone')
                st.plotly_chart(fig, use_column_width=True)
        else:
            st.info("No data available")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <p>🎨 AI-Powered Design Studio</p>
        <p>© {datetime.now().year} {COMPANY_NAME} | All Rights Reserved</p>
        <p style="font-size: 0.8rem;">☁️ Cloud Backup: Supabase + Cloudinary</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()