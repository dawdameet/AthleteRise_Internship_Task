import streamlit as st
import os
import shutil
import base64
from cover_drive_analysis_realtime import video_dekho
import json



#HELPERS (DOWNLOADER)
def get_binary_file_downloader_html(file_path, file_label):
    """Generates a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'
    return href


# UI FILE
st.set_page_config(page_title="ATHLETERISE", layout="wide")

st.title("AthleteRise: Cover Drive Analyzer")
st.markdown("Upload a video of a cover drive to get biomechanical analysis and feedback.")

# TEMP OUTPUT
temp_dir = "./temp_output"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

#UPLOADER
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # SAVE TO TEMP
    input_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # THIS WILL BE SHOWING THE VIDEO IN THE WEBSIE
    st.subheader("Uploaded Video")
    st.video(input_video_path)
    
    st.markdown("---")
    
    if st.button("Start Analysis"):
        # WE NEED SEPERATE DIR FOR THIS RUN
        processed_output_dir = os.path.join(temp_dir, "processed")
        if os.path.exists(processed_output_dir):
            shutil.rmtree(processed_output_dir)
        os.makedirs(processed_output_dir)
        
        with st.spinner("Processing video... This may take a few moments."):
            try:
                #PLAY THE VIDEO DEKHO
                video_dekho(input_video_path, output_folder=processed_output_dir)
                
                st.success("Analysis complete!")
                
                # EXTRACT THE PATH WHERE THE PROCESSING HAPPENED
                annotated_video_path = os.path.join(processed_output_dir, "annotated_video.mp4")
                json_report_path = os.path.join(processed_output_dir, "evaluation.json")
                plot_image_path = os.path.join(processed_output_dir, "temporal_analysis.png")

                # SHOW THE PROCEESSED VDO
                st.subheader("Processed Video Playback")
                st.video(annotated_video_path)

                st.subheader("Analysis Results")
                
                # SHOW JSON (BEAUTIFUL)
                if os.path.exists(json_report_path):
                    with open(json_report_path, 'r') as f:
                        report_data = json.load(f)
                    st.json(report_data)

                # WHERE ARE THE TEMPORAL PLOTS
                if os.path.exists(plot_image_path):
                    st.image(plot_image_path, caption="Temporal Biomechanical Analysis")

                st.subheader("Downloads")
                # MAKE DOWNLOADABLE
                if os.path.exists(annotated_video_path):
                    st.markdown(get_binary_file_downloader_html(annotated_video_path, 'Download Annotated Video'), unsafe_allow_html=True)
                
                if os.path.exists(json_report_path):
                    st.markdown(get_binary_file_downloader_html(json_report_path, 'Download JSON Report'), unsafe_allow_html=True)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Please ensure your video is in a supported format and contains a clear view of the subject.")