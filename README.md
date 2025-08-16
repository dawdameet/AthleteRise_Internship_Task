# AthleteRise: AI-Powered Cover Drive Analysis

This project is a Python-based system designed to perform real-time biomechanical analysis of a cricket cover drive from a full video recording. The application processes video input frame-by-frame to extract pose estimation data, calculate key performance metrics, and generate a comprehensive evaluation of the player's technique.

The system is delivered as an interactive Streamlit web application for ease of use.

## Features

This project fulfills all base requirements of the assignment and implements several advanced features.

### Core Features

  * **Full Video Processing**: The system processes video files sequentially from start to finish, generating a single, annotated output video.
  * **Pose Estimation**: It uses the MediaPipe library to extract 33 key body landmarks for each frame, with graceful handling of occluded or undetected joints.
  * **Biomechanical Metrics**: The following metrics are computed and tracked in real-time:
      * Front Elbow Angle
      * Spine Lean
      * Head-over-Knee Alignment
      * Front Foot Angle
  * **Live Overlays**: The output video includes a rendered pose skeleton, live metric readouts, and simple, color-coded feedback cues to provide immediate visual feedback.
  * **Final Evaluation Report**: Upon completion, a `evaluation.json` file is generated, containing a 1-10 score and actionable feedback across five categories: Footwork, Head Position, Swing Control, Balance, and Follow-through.

### Implemented Advanced Features

  * **Automatic Phase Segmentation**: The system automatically detects the distinct phases of the shot (Stance, Downswing, Impact, Follow-Through) by analyzing wrist velocity.
  * **Impact Moment Detection**: The point of bat-on-ball contact is identified by detecting the peak in wrist velocity during the downswing.
  * **Temporal Smoothness Analysis**: Motion consistency is evaluated by analyzing frame-to-frame changes in angles. A summary chart (`temporal_analysis.png`) is generated to visualize metric stability over time.
  * **Basic Bat Detection**: The bat's position and swing path are approximated by drawing a vector from the batter's wrists.
  * **Streamlit Web Application**: A simple, interactive UI allows users to upload a video, run the analysis, view the processed playback, and download the resulting report files.

-----

## Technology Stack

  * **Python 3.8+**
  * **Streamlit**: For the web interface.
  * **OpenCV**: For video processing and frame handling.
  * **MediaPipe**: For high-fidelity pose estimation.
  * **NumPy**: For numerical calculations.
  * **Matplotlib**: For generating the temporal analysis plot.

-----

## Setup and Usage

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install Dependencies**:
    It's recommended to use a virtual environment. The required packages can be installed from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application**:
    Launch the Streamlit app from your terminal.
    ```bash
    streamlit run app.py
    ```
4.  **Upload and Analyze**:
    Open the provided local URL in your web browser, upload a video of a cover drive, and click "Start Analysis" to begin processing.

-----

## Project Structure

  * `app.py`: The main Streamlit application file that handles the UI and file I/O.
  * `cover_drive_analysis_realtime.py`: The core module containing all logic for video processing, pose estimation, metric calculation, and report generation.
  * `/output/`: The default directory where all generated files are saved.
  * `requirements.txt`: A list of all Python dependencies.
  * `README.md`: Project documentation.

-----

## Assumptions and Limitations

  * **Viewpoint**: The analysis assumes a side-on or slightly angled view of the batter for optimal landmark detection.
  * **Ball Detection**: The current implementation uses a simple HSV color mask to detect a red ball. Its effectiveness may vary with different ball colors or lighting conditions.
  * **Bat Tracking**: The bat is approximated as a line extending from the wrists and is not detected as a discrete object. This approach is sufficient for estimating the swing path but not for detailed bat-specific analytics.