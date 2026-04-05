import streamlit as st
import os
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from detection import YOLODetector
from retrieval import RetrievalPipeline

st.set_page_config(page_title="Visual Product Search", layout="wide")

@st.cache_resource
def load_detector():
    return YOLODetector()

@st.cache_resource
def load_pipeline(index_path, alpha, clip_model):
    if not os.path.exists(index_path):
        return None
    return RetrievalPipeline(index_path=index_path, alpha=alpha, clip_model_path=clip_model)

def main():
    st.title("👗 Visual Product Search Engine")
    st.markdown("Upload a fashion image to find visually similar items from our catalog!")

    st.sidebar.header("Settings")
    k_val = st.sidebar.slider("Number of results (K)", min_value=5, max_value=30, value=10, step=5)
    alpha_val = st.sidebar.slider("Alpha (Vision weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    index_path = st.sidebar.text_input("HNSW Index Path", value="./index/hnsw_index")
    clip_model = st.sidebar.text_input("CLIP Model Path (Optional)", value="./models/clip_finetuned/clip_finetuned_seed_42.pt")
    
    if not os.path.exists(clip_model):
        clip_model = None

    uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        detector = load_detector()
        
        if 'cropped_img' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
            st.session_state['cropped_img'] = detector.crop_primary_item(image)
            st.session_state['last_upload'] = uploaded_file.name
            
        with col2:
            st.subheader("Detected Item")
            st.image(st.session_state['cropped_img'], use_container_width=True)
            
            if st.button("Re-crop (Use Original)"):
                st.session_state['cropped_img'] = image
                st.rerun()

            confirm_search = st.button("Confirm Crop & Search")

        if confirm_search:
            with st.spinner("Retrieving similar products..."):
                pipeline = load_pipeline(index_path, alpha_val, clip_model)
                if pipeline is None:
                    st.error("Index not found. Please run the offline indexing pipeline first.")
                    return
                    
                pipeline.k = k_val
                results = pipeline.retrieve(st.session_state['cropped_img'])
                
                st.success(f"Found top {k_val} results!")
                
                # Display Results
                cols = st.columns(5)
                for idx, res in enumerate(results[:k_val]):
                    with cols[idx % 5]:
                        try:
                            res_img = Image.open(res['image_path'])
                            st.image(res_img, use_container_width=True, caption=f"ID: {res['item_id']}")
                        except:
                            st.write(f"Image not found. ID: {res['item_id']}")
                        st.caption(f"Score: {res.get('itm_score', res.get('ann_score', 0)):.3f}")

if __name__ == "__main__":
    main()
