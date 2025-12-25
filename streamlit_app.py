import streamlit as st
import requests
import io
import base64
from PIL import Image
import json
import tempfile
import cv2
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Crowd Counting App",
    page_icon="👥",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this for production

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_crowd_count(image_file):
    """Send image to API for prediction"""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=20)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    st.title("👥 Crowd Counting Application")
    st.markdown("Upload an image to estimate the crowd count using AI")
    
    # Check API status
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if check_api_health():
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")
            st.warning("Please start the FastAPI server first")
            return
    
    # Main interface
    st.markdown("---")
    

    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis"])

    with tab1:
        st.subheader("📤 Upload Image")
        uploaded_files = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload an image containing people/crowds to count"
        )


        if uploaded_files is not None:
            for i, uploaded_file in enumerate(uploaded_files):
            # Display uploaded image
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🖼️ Input Image")
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    # Image info
                    st.info(f"**Filename:** {uploaded_file.name}")
                    st.info(f"**Size:** {image.size[0]} x {image.size[1]} pixels")

                with col2:
                    st.subheader("🔄 Processing")

                    if st.button(f"🚀 Analyze Image {i+1}", type="primary", use_container_width=True, key=f"analyze_{i}"):
                        with st.spinner("Analyzing image... This may take a few moments."):
                            # Reset file pointer
                            uploaded_file.seek(0)

                            # Get prediction
                            result = predict_crowd_count(uploaded_file)

                            if result and result.get("success"):
                                st.success("✅ Analysis Complete!")

                                # Display results
                                st.markdown("---")
                                st.subheader("📊 Results")

                                # Count result
                                count = result.get("estimated_count", 0)
                                st.metric(
                                    label="Estimated Crowd Count",
                                    value=f"{count:,.0f}",
                                    help="Number of people detected in the image"
                                )

                                # Model info
                                model_info = result.get("model_info", {})
                                st.info(f"**Model:** {model_info.get('model_name', 'Unknown')} "
                                       f"({model_info.get('model_weights', 'Unknown')} weights)")

                                # Density map
                                if result.get("density_map"):
                                    st.subheader("🌡️ Density Map")
                                    try:
                                        # Decode base64 image
                                        density_map_data = result["density_map"].split(",")[1]
                                        density_map_bytes = base64.b64decode(density_map_data)
                                        density_map_image = Image.open(io.BytesIO(density_map_bytes))

                                        st.image(
                                            density_map_image,
                                            caption="Crowd Density Visualization (Red = Higher Density)",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Could not display density map: {e}")

                                # Download results
                                st.markdown("---")
                                st.subheader("💾 Download Results")

                                result_data = {
                                    "filename": uploaded_file.name,
                                    "estimated_count": count,
                                    "model": model_info
                                }
    
                                st.download_button(
                                    label="📄 Download Results (JSON)",
                                    data=json.dumps(result_data, indent=2),
                                    file_name=f"crowd_count_{uploaded_file.name.split('.')[0]}.json",
                                    mime="application/json"
                                )
                            else:
                                st.error("❌ Failed to analyze image. Please try again.")

    with tab2:
        st.subheader("Upload Video for Trend Analysis")
        video_file = st.file_uploader("Choose a video", type=['mp4', 'mov', 'avi'])
        
        interval = st.slider("Analysis Interval (seconds)", min_value=5, max_value=60, value=30)
        
        if video_file:
            # 1. Show the Media Player immediately
            st.markdown("### 📺 Preview")
            st.video(video_file)  # <--- This adds the playable video player
            
            # Save video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            st.info(f"Video Duration: {duration:.2f}s | FPS: {fps}")
            
            if st.button("🚀 Start Video Analysis"):
                frames_to_process = int(fps * interval)
                results_list = []
                
                # Layout for Live Status
                st.markdown("---")
                st.subheader("⚡ Live Analysis")
                
                # 2. Create columns to show the Chart and the Current Frame side-by-side
                col_chart, col_img = st.columns([2, 1])
                
                with col_chart:
                    chart_placeholder = st.empty()
                
                with col_img:
                    frame_placeholder = st.empty() # <--- Placeholder for the processed frame
                
                progress_bar = st.progress(0)
                
                current_frame = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if current_frame % frames_to_process == 0:
                        # Convert to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        # 3. Update the processed frame view
                        frame_placeholder.image(
                            pil_img, 
                            caption=f"Analyzing Frame at {current_frame/fps:.1f}s", 
                            use_container_width=True
                        )
                        
                        # Prepare for API
                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        
                        try:
                            # API Call
                            files = {"file": ("frame.jpg", img_byte_arr, "image/jpeg")}
                            response = requests.post(f"{API_BASE_URL}/predict", files=files)
                            
                            if response.status_code == 200:
                                data = response.json()
                                count = data.get("estimated_count", 0)
                                timestamp = current_frame / fps
                                results_list.append({"Time (s)": round(timestamp, 1), "Count": count})
                                
                                # Update Chart
                                df = pd.DataFrame(results_list)
                                chart_placeholder.line_chart(df.set_index("Time (s)"))
                                
                        except Exception as e:
                            st.error(f"Error: {e}")
                            
                    current_frame += 1
                    progress_bar.progress(min(current_frame / frame_count, 1.0))
                
                cap.release()
                st.success("✅ Analysis Complete")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses state-of-the-art deep learning models to count people in images.
        
        **Features:**
        - Real-time crowd counting
        - Density map visualization
        - Multiple model support
        - Easy-to-use interface
        
        **Supported Models:**
        - DM-Count (Default)  
        - CSRNet
        - SFANet
        - Bayesian Counting
        """)
        
        st.header("🔧 Usage Tips")
        st.markdown("""
        - Upload clear images with visible people.
        - Works best with crowd scenes.
        - Processing may take 10-30 seconds.
        - Larger images take longer to process.
        """)
        
        # st.header("📈 Model Performance")
        # st.markdown("""
        # **Typical Accuracy:**
        # - Dense crowds: ±60-70 people
        # - Sparse crowds: ±8-12 people
        # - Overall correlation: >0.85
        # """)

if __name__ == "__main__":
    main()
