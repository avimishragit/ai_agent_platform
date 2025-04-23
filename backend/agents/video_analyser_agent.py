import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import tempfile
from pathlib import Path
import logging

def configure_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def configure_gemini():
    """Configure Gemini API with environment variables"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    return api_key

@st.cache_resource
def create_agent():
    """Create and configure Gemini AI agent"""
    return Agent(
        model=Gemini(id="gemini-1.5-flash", temperature=0.3),
        markdown=True,
        show_tool_calls=False
    )

def process_video(uploaded_file):
    """Handle video file upload and temporary storage"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def generate_prompt(user_query):
    """Generate structured prompt for video analysis"""
    return f"""
**Role**: Expert Video Analyst specializing in cultural context and emotional intelligence

**Task**: Analyze the video content and respond to: "{user_query}"

**Output Requirements**:
- Use Markdown formatting with clear section headers
- Include timestamped analysis where applicable
- List key observations in bullet points
- Highlight cultural context considerations
- Identify emotional cues from speech/visuals

**Guidelines**:
1. Maintain factual accuracy and cultural sensitivity
2. Focus on actionable insights over generic descriptions
3. Adhere to Indian legal and cultural norms
4. Limit response to 500 words
5. Avoid technical jargon and disclaimers

**Analysis Framework**:
1. Content Summary (3-5 bullet points)
2. Emotional Analysis (voice tonality, facial expressions)
3. Cultural Context Observations
4. Key Themes and Patterns
5. Potential Applications/Considerations
"""

def analyze_video(video_path, user_query, agent):
    """Handle video analysis workflow"""
    start_time = time.time()
    
    try:
        video_content = genai.upload_file(video_path)
        while video_content.state.name == "PROCESSING":
            time.sleep(1)
            video_content = genai.get_file(video_content.name)
        
        response = agent.run(generate_prompt(user_query), videos=[video_content])
        return {
            "content": response.content,
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

def setup_ui():
    """Configure Streamlit UI components"""
    st.set_page_config(
        page_title="Video Insights Analyzer",
        page_icon="📽️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.title("Video Insights Analyzer")
    st.markdown("**Powered by Gemini AI**")

def main():
    """Main application workflow"""
    logger = configure_logging()
    
    try:
        configure_gemini()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        st.error("Failed to initialize API configuration")
        st.stop()

    setup_ui()
    video_agent = create_agent()

    uploaded_file = st.file_uploader(
        "Select video file",
        type=["mp4", "mov", "avi"],
        help="Supported formats: MP4, MOV, AVI (max 100MB)"
    )

    if uploaded_file:
        tmp_path = process_video(uploaded_file)
        
        try:
            st.video(tmp_path)
            user_query = st.text_area(
                "Analysis Request",
                value="Provide a comprehensive analysis of the video content including emotional cues and cultural context.",
                height=100
            )

            if st.button("Analyze Content", type="primary"):
                with st.spinner("Analyzing video content..."):
                    try:
                        result = analyze_video(tmp_path, user_query, video_agent)
                        
                        st.subheader("Analysis Report")
                        st.markdown(result["content"])
                        st.caption(f"Analysis completed in {result['processing_time']:.2f} seconds")
                        
                    except Exception as e:
                        st.error("Failed to generate analysis")
                        logger.error(f"Analysis error: {str(e)}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        st.info("Please upload a video file to begin analysis")

if __name__ == "__main__":
    main()
