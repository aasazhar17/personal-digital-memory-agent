import os
import sys
from dotenv import load_dotenv

# Set up module resolution path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables (e.g. GEMINI_API_KEY) if available in a local .env file
load_dotenv()

from ui.streamlit_ui import render_ui

if __name__ == "__main__":
    # Renders and runs the main Streamlit application
    render_ui()
