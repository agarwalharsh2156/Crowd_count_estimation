import streamlit as st
import requests
import io
import base64
from PIL import Image
import json

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
    
    # File upload section
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
                st.image(image, caption="Uploaded Image", use_column_width=True)

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
                                        use_column_width=True
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
