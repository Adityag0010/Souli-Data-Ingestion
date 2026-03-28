"""
Data Ingestion UI - Upload CSV with YouTube links and process videos
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import asyncio
import tempfile
from souli_pipeline.config_loader import load_config
from souli_pipeline.utils.run_id import get_run_id
from souli_pipeline.youtube.pipeline import run_youtube_pipeline

def show():
    st.header("🎬 Data Ingestion Interface")
    st.write("""
    Upload a CSV file with YouTube links to process multiple videos.
    The CSV should have a column named **url** or **youtube_url**.
    """)
    
    # Instructions
    with st.expander("📋 CSV Format & Instructions", expanded=False):
        st.markdown("""
        ### Expected CSV Format:
        | url | name | title |
        |----------|------|-------|
        | https://youtu.be/VIDEO_ID_1 | video_1 | My Video 1 |
        | https://youtu.be/VIDEO_ID_2 | video_2 | My Video 2 |
        
        **Columns:**
        - **yt_links** (required): YouTube URL
        - **name** (optional): Video name for folder
        - **title** (optional): Display title for output tracking
        
        **Download example CSV:**
        """)
        st.download_button(
            label="📥 Download Example CSV",
            data=create_example_csv(),
            file_name="sample_youtube_videos.csv",
            mime="text/csv"
        )
    
    # Config selection
    col1, col2 = st.columns(2)
    with col1:
        config_path = st.text_input(
            "Config file path",
            value="configs/pipeline.yaml",
            help="Path to your pipeline configuration file"
        )
    
    with col2:
        # Check if config exists
        if os.path.exists(config_path):
            st.success("✅ Config file found")
        else:
            st.warning(f"⚠️ Config not found at: {config_path}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV with YouTube links",
        type=["csv"],
        help="CSV file containing YouTube video links"
    )
    
    if uploaded_file is not None:
        # Read and display CSV
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df, use_container_width=True)
        
        # Validate CSV
        if not validate_csv(df):
            st.error("❌ CSV must have 'url' or 'youtube_url' column")
            return
        
        st.success("✅ CSV is valid")
        
        # Processing options
        st.write("### Processing Options:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            skip_tagging = st.checkbox(
                "Skip energy node tagging",
                value=False,
                help="Skip Qwen energy node tagging (faster)"
            )
        
        with col2:
            merge_outputs = st.checkbox(
                "Merge outputs",
                value=True,
                help="Merge all video outputs into single files"
            )
        
        with col3:
            output_format = st.selectbox(
                "Output format",
                ["xlsx", "json", "both"],
                help="Format for output files"
            )
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            process_videos(df, config_path, skip_tagging, merge_outputs, output_format)
    
    # Previous runs section
    st.write("---")
    st.write("### 📊 Previous Runs")
    display_previous_runs()


def validate_csv(df):
    """Validate CSV has required columns"""
    required_cols = {"yt_links", "youtube_url", "url"}
    actual_cols = set(df.columns)
    return bool(actual_cols & required_cols)


def create_example_csv():
    """Create example CSV data"""
    example_data = {
        "url": [
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/jNQXAC9IVRw"
        ],
        "name": ["video_1", "video_2"],
        "title": ["Sample Video 1", "Sample Video 2"]
    }
    df = pd.DataFrame(example_data)
    return df.to_csv(index=False)


def process_videos(df, config_path, skip_tagging, merge_outputs, output_format):
    """Process videos from CSV"""
    
    if not os.path.exists(config_path):
        st.error(f"Config file not found: {config_path}")
        return
    
    try:
        cfg = load_config(config_path)
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return
    
    # Create progress bars
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    total_videos = len(df)
    successful = 0
    failed = 0
    
    try:
        rid = get_run_id()
        
        for idx, row in df.iterrows():
            video_url = row.get("yt_links") or row.get("youtube_url") or row.get("url")
            video_name = row.get("name", f"video_{idx+1}")
            
            if not video_url:
                st.warning(f"Row {idx+1}: No URL found, skipping")
                failed += 1
                continue
            
            # Update status
            status_text.write(f"Processing video {idx+1}/{total_videos}: {video_name}")
            
            try:
                # Run pipeline
                out_dir = os.path.join(cfg.run.outputs_dir, rid, "youtube", f"video_{idx+1:03d}")
                os.makedirs(out_dir, exist_ok=True)
                
                output = run_youtube_pipeline(
                    cfg,
                    youtube_url=video_url,
                    out_dir=out_dir,
                    tag_energy=not skip_tagging
                )
                
                successful += 1
                
                with results_container:
                    st.success(f"✅ {video_name} processed successfully")
                    with st.expander("View details"):
                        for k, v in output.items():
                            st.write(f"- {k}: {v}")
                
            except Exception as e:
                failed += 1
                with results_container:
                    st.error(f"❌ {video_name} failed: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / total_videos
            progress_bar.progress(progress)
        
        # Final summary
        st.write("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Videos", total_videos)
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Failed", failed)
        
        st.success(f"✅ Processing complete! Run ID: `{rid}`")
        st.info(f"📁 Outputs saved to: `{cfg.run.outputs_dir}/{rid}/`")
        
    except Exception as e:
        st.error(f"Fatal error during processing: {str(e)}")


def display_previous_runs():
    """Display previous runs from outputs directory"""
    outputs_dir = "outputs"
    
    if not os.path.exists(outputs_dir):
        st.info("No previous runs found")
        return
    
    runs = []
    try:
        for run_id in os.listdir(outputs_dir):
            run_path = os.path.join(outputs_dir, run_id)
            if os.path.isdir(run_path):
                runs.append(run_id)
    except Exception as e:
        st.error(f"Error reading runs: {str(e)}")
        return
    
    if not runs:
        st.info("No previous runs found")
        return
    
    selected_run = st.selectbox("View previous run:", sorted(runs, reverse=True))
    
    if selected_run:
        run_path = os.path.join(outputs_dir, selected_run)
        youtube_path = os.path.join(run_path, "youtube")
        
        if os.path.exists(youtube_path):
            st.write(f"**Run ID:** {selected_run}")
            
            # List output files
            try:
                files = os.listdir(youtube_path)
                if files:
                    st.write("**Output Files:**")
                    for file in sorted(files):
                        file_path = os.path.join(youtube_path, file)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        st.write(f"- `{file}` ({file_size:.2f} KB)")
                else:
                    st.info("No output files found")
            except Exception as e:
                st.error(f"Error listing files: {str(e)}")