# SO THIS IS THE PYTHON SCRIPT 
# WHERE WE WILL BE EXTRACTING THE SHOT INTERNALS
# THIS WAS BUILT UPON THE MEDIAPIE AND OPENCV 
# AUTHOR: MEET DAWDA

# I WILL WALK YOU THROUGH ALL THE SECTIONS OF THIS CODE

import os 
import cv2 
import time 
import numpy as np
import math

# ENSURE PYTHON 3.8.^ 
# I AM USING PYTHON  3.8.0
import mediapipe as mp 

import json
import matplotlib.pyplot as plt 

# ALRIGHT SO THESE ARE JUST NUMBERS FOR THE DIFFERENT BODY PARTS THAT MEDIAPIPE GIVES US
# INSTEAD OF REMEMBERING THAT THE NOSE IS 0 WE CAN JUST WRITE NOSE MUCH EASIER
# I GOT ALL THESE FROM THE OFFICIAL MEDIAPIPE PICTURE
MARKS = {
    "NOSE": 0, "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14, "LEFT_WRIST": 15, 
    "RIGHT_WRIST": 16, "LEFT_THUMB": 22, "LEFT_PINKY": 17,
    "RIGHT_THUMB": 19, "RIGHT_INDEX": 20, "RIGHT_PINKY": 18,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26, "LEFT_ANKLE": 27, 
    "RIGHT_ANKLE": 28, "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32
}

# THIS FUNCTION IS ALL ABOUT CALCULATING AN ANGLE
# YOU GIVE IT THREE POINTS SAY X Y AND Z IT WILL CALCULATE THE
# ANGLE AT POINT Y SO FOR AN ELBOW YOUD PASS IN THE SHOULDER ELBOW AND WRIST
def calc_angle(x, y, z):
    # IF ANY OF THE POINTS ARE MISSING WE CANT DO MATH SO WE JUST GIVE UP
    if x is None or y is None or z is None:
        return None
    
    # THIS IS A BIT OF VECTOR MATH WERE BASICALLY FINDING THE ANGLE
    # BETWEEN THE LINE FROM Y TO X AND THE LINE FROM Y TO Z
    v1 = x - y
    v2 = z - y
    v1_hat = v1 / np.linalg.norm(v1)
    v2_hat = v2 / np.linalg.norm(v2)
    ang = np.arccos(np.dot(v1_hat, v2_hat))
    
    # THE RESULT IS IN RADIANS BUT DEGREES ARE EASIER FOR US HUMANS TO UNDERSTAND
    return np.degrees(ang)

# AND THIS ONE CALCULATES THE ANGLE OF A LINE COMPARED TO THE FLAT HORIZONTAL GROUND
def ang_to_x(m):
    # AGAIN IF WE DONT HAVE THE POINT WE CANT DO ANYTHING
    if m is None: return None
    
    # MORE VECTOR MATH THIS TIME WERE COMPARING OUR LINE M TO A HORIZONTAL LINE WHICH IS 1 0
    m_hat = m / np.linalg.norm(m)
    ang = np.arccos(np.dot(m_hat, [1, 0]))
    return np.degrees(ang)

# THIS IS A CLEVER LITTLE FUNCTION TO FIGURE OUT WHICH FOOT IS THE FRONT FOOT
# FOR A RIGHT HANDED BATTER THE LEFT FOOT IS IN FRONT AND VICE VERSA
# IT WORKS BY SEEING WHICH KNEE IS FURTHER OUT FROM THE CENTER OF THE HIPS
def extract_front_foot(lk, rk, lh, rh):
    # IF WERE MISSING ANY OF THE KEY BODY PARTS WELL JUST GUESS ITS A RIGHTY
    if lk is None or rk is None or lh is None or rh is None:
        return "LEFT" 
    
    hip_midx = 0.5 * (lh[0] + rh[0])
    left_front_gap = abs(lk[0] - hip_midx)
    right_front_gap = abs(rk[0] - hip_midx)
    
    # WHICHEVER KNEE HAS A BIGGER GAP IS PROBABLY THE FRONT ONE
    return "LEFT" if left_front_gap > right_front_gap else "RIGHT"

# A SIMPLE HELPER TO GET THE ACTUAL SCREEN COORDINATES IN PIXELS FROM MEDIAPIPES RESULTS
# MEDIAPIPE GIVES COORDINATES AS A PERCENTAGE OF THE SCREEN SO WE HAVE TO MULTIPLY BY WIDTH AND HEIGHT
def get_coordinates(landmarks_list, w, h, ix):
    try:
        l = landmarks_list.landmark[ix]
        # IF MEDIAPIPE ISNT VERY CONFIDENT IT CAN SEE THE LANDMARK WELL JUST IGNORE IT
        if l.visibility is not None and l.visibility < 0.5:
            return None
        return np.array([l.x * w, l.y * h], dtype=np.float32)
    except Exception:
        return None

# WHEN WERE DISPLAYING LIVE DATA ON THE SCREEN SOMETIMES A LANDMARK DISAPPEARS FOR A FRAME
# THIS FUNCTION JUST GRABS THE MOST RECENT VALID NUMBER FROM OUR LIST SO THE DISPLAY DOESNT LOOK GLITCHY
def get_last_valid(data_list):
    for val in reversed(data_list):
        if val is not None and not np.isnan(val):
            return val
    return None

# JUST A FUNCTION TO DRAW TEXT ON THE VIDEO
# IT DRAWS THE TEXT TWICE ONCE IN BLACK AND A BIT THICKER AND THEN IN THE ACTUAL COLOR
# THIS CREATES A NICE LITTLE OUTLINE EFFECT WHICH MAKES IT SUPER EASY TO READ ON ANY BACKGROUND
def put_text(frame, text, pos, scale=0.6, color=(255,255,255), thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

# TRYING TO FIND THE EXACT MOMENT THE BAT HITS THE BALL
# THE IDEA IS THAT THE BAT IS MOVING FASTEST AT THE POINT OF IMPACT
# SO WE JUST LOOK FOR THE FRAME WITH THE HIGHEST WRIST VELOCITY
def find_impact_moment(velocities):
    if not velocities or np.all(np.isnan(velocities)):
        return None
    return int(np.nanargmax(velocities))

# THIS CALCULATES THE CHANGE IN A VALUE FROM ONE FRAME TO THE NEXT
# ITS USEFUL FOR CHECKING THE SMOOTHNESS OF A MOTION A JERKY MOTION
# WILL HAVE BIG CHANGES WHILE A SMOOTH ONE WILL HAVE SMALL CHANGES
def calculate_deltas(data):
    """CALCULATES THE ABSOLUTE FRAMETOFRAME CHANGE FOR A METRIC"""
    if len(data) < 2:
        return []
    
    valid_data = np.array([x for x in data if x is not None and not np.isnan(x)])
    if len(valid_data) < 2:
        return []
        
    return np.abs(np.diff(valid_data))

# THIS PART IS FOR CREATING THE GRAPHS AT THE END OF THE ANALYSIS
# IT JUST MAKES A COUPLE OF PLOTS TO SHOW HOW THE ANGLES AND SPEEDS CHANGED OVER THE WHOLE SHOT
def generate_plots(metrics_data, output_folder):
    timestamps = np.arange(len(metrics_data["elbow_angles"]))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # THE FIRST PLOT SHOWS THE MAIN ANGLES AND VELOCITIES OVER TIME
    ax1.plot(timestamps, metrics_data["elbow_angles"], label="Elbow Angle")
    ax1.plot(timestamps, metrics_data["spine_leans"], label="Spine Lean")
    ax1.plot(timestamps, metrics_data["wrist_velocities"], label="Wrist Velocity")
    ax1.set_title("Biomechanical Metrics Over Time")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    
    # THE SECOND PLOT IS FOR SMOOTHNESS SHOWING THE FRAMEBYFRAME CHANGES
    elbow_deltas = np.insert(calculate_deltas(metrics_data["elbow_angles"]), 0, np.nan)
    spine_deltas = np.insert(calculate_deltas(metrics_data["spine_leans"]), 0, np.nan)
    
    # THIS BIT IS JUST TO MAKE SURE THE DATA ARRAYS ARE THE RIGHT SIZE FOR PLOTTING SOMETIMES THEY CAN BE OFF BY ONE
    if len(elbow_deltas) != len(timestamps):
        pad_count = len(timestamps) - len(elbow_deltas)
        elbow_deltas = np.pad(elbow_deltas, (0, pad_count), 'constant', constant_values=np.nan)
    
    if len(spine_deltas) != len(timestamps):
        pad_count = len(timestamps) - len(spine_deltas)
        spine_deltas = np.pad(spine_deltas, (0, pad_count), 'constant', constant_values=np.nan)
        
    ax2.plot(timestamps, elbow_deltas, label="Elbow Delta")
    ax2.plot(timestamps, spine_deltas, label="Spine Delta")
    ax2.set_title("Smoothness (Frame-to-Frame Changes)")
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Absolute Angle Change (deg)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, "temporal_analysis.png")
    plt.savefig(plot_path)
    print(f"Temporal plot saved to {plot_path}")
    plt.close()

# THIS IS WHERE ALL THE REAL SHI HAPPENS FOR THE FINAL REPORT
# IT TAKES ALL THE DATA WE COLLECTED CALCULATES AVERAGES AND SCORES THE TECHNIQUE
def final_evaluation(metrics_data, cfg, output_folder="./output"):
    elbow_angles = metrics_data["elbow_angles"]
    spine_leans = metrics_data["spine_leans"]
    head_over_knee_offsets = metrics_data["head_over_knee_offsets"]
    foot_angles = metrics_data["foot_angles"]
    
    # WE CALCULATE THE AVERAGE FOR EACH METRIC IGNORING ANY FRAMES WHERE WE COULDNT SEE THE BODY PART
    avg_elbow = np.nanmean(elbow_angles).item() if elbow_angles else None
    avg_spine = np.nanmean(spine_leans).item() if spine_leans else None
    avg_hok = np.nanmean(head_over_knee_offsets).item() if head_over_knee_offsets else None
    avg_foot = np.nanmean(foot_angles).item() if foot_angles else None
    
    elbow_deltas = calculate_deltas(elbow_angles)
    spine_deltas = calculate_deltas(spine_leans)
    
    # HERE WE CHECK THE STANDARD DEVIATION OF THE DELTAS A SMALLER NUMBER MEANS A SMOOTHER MOTION
    elbow_smoothness = np.nanstd(elbow_deltas).item() if elbow_deltas.size > 0 else np.nan
    spine_smoothness = np.nanstd(spine_deltas).item() if spine_deltas.size > 0 else np.nan

    # A SIMPLE SCORING SYSTEM IF YOUR VALUE IS WITHIN THE GOOD RANGE YOU GET A HIGH SCORE
    # THE FURTHER YOU ARE OUTSIDE THE RANGE THE LOWER THE SCORE
    def score_metric(val, min_val, max_val, factor):
        # IF WE HAVE NO DATA GIVE A LOW SCORE
        if val is None: return 3.0 
        dist = 0
        if val < min_val: dist = min_val - val
        elif val > max_val: dist = val - max_val
        return max(1.0, 9.0 - dist * factor)
    
    # JUST CONVERTS A SCORE INTO A WORD RATING
    def feedback_for_score(score):
        if score >= 8.5: return "Excellent"
        if score >= 6.5: return "Good"
        if score >= 4.5: return "Fair"
        return "Needs Work"
    
    # SCORE EACH PART OF THE TECHNIQUE
    footwork_score = score_metric(avg_foot, 0, cfg["front_foot_angle_max_good"], 0.2)
    head_pos = score_metric(avg_hok, 0, cfg["head_over_knee_max_px"], 0.1)
    swing_ctrl = score_metric(avg_elbow, cfg["front_elbow_good_min"], cfg["front_elbow_good_max"], 0.05)
    balance = score_metric(avg_spine, 0, cfg["spine_lean_max_good"], 0.15)
    
    # FOLLOW THROUGH IS KIND OF A MIX OF GOOD SWING AND GOOD BALANCE
    fts = (swing_ctrl + balance) / 2
    
    # NOW WE JUST PUT ALL THE SCORES AND FEEDBACK INTO A DICTIONARY
    report = {
        "summary": {
            "footwork": round(footwork_score, 1),
            "head_position": round(head_pos, 1),
            "swing_control": round(swing_ctrl, 1),
            "balance": round(balance, 1),
            "follow_through": round(fts, 1),
        },
        "smoothness": {
            "elbow_smoothness_std": round(elbow_smoothness, 2) if not np.isnan(elbow_smoothness) else None,
            "spine_smoothness_std": round(spine_smoothness, 2) if not np.isnan(spine_smoothness) else None,
            "feedback": "Smoothness is key to power and control A lower standard deviation indicates a more fluid motion"
        },
        "feedback": {
            "footwork": [
                f"{feedback_for_score(footwork_score)} footwork Average angle was {avg_foot:.1f} degrees",
                "Focus on planting your front foot at a zero-degree angle to the crease"
            ],
            "head_position": [
                f"{feedback_for_score(head_pos)} head position Average offset was {avg_hok:.1f} pixels",
                "Try to keep your head perfectly still and stacked over your front knee"
            ],
            "swing_control": [
                f"{feedback_for_score(swing_ctrl)} swing control Average elbow angle was {avg_elbow:.1f} degrees",
                "Work on maintaining your V shape with the front elbow throughout the shot"
            ],
            "balance": [
                f"{feedback_for_score(balance)} balance Average spine lean was {avg_spine:.1f} degrees",
                "Keep your back as straight as possible to avoid losing balance and power"
            ],
            "follow_through": [
                f"{feedback_for_score(fts)} follow-through A strong follow-through is key to power",
                "Let your body momentum carry the bat through after impact for a clean finish"
            ]
        }
    }
    # FINALLY SAVE THE REPORT AS A JSON FILE SO ANOTHER PROGRAM COULD READ IT EASILY
    out_path = os.path.join(output_folder, "evaluation.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nFinal evaluation saved to {out_path}")

# THIS IS A SIMPLE BALL DETECTOR IT JUST LOOKS FOR ANYTHING THATS RED
# ITS NOT PERFECT BUT ITS GOOD ENOUGH FOR A START
def detect_ball(frame):
    """DETECTS A RED CRICKET BALL USING COLOR FILTERING"""
    # ITS EASIER TO FIND COLORS IN HSV FORMAT THAN IN THE NORMAL BGR FORMAT
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # THESE VALUES DEFINE THE RANGE OF RED THAT WERE LOOKING FOR
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # WE ASSUME THE BIGGEST RED THING IS THE BALL
        biggest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(biggest_contour)
        # WE CHECK THE RADIUS TO MAKE SURE WERE NOT PICKING UP SOME OTHER TINY RED SPECK
        if radius > 5 and radius < 20: 
            return (int(x), int(y))
    return None


# THIS FUNCTION TRIES TO GUESS WHERE THE BAT IS
# WE DONT HAVE A LANDMARK FOR THE BAT SO WE JUST DRAW A LINE
# THE LINE STARTS AT THE FRONT WRIST AND POINTS IN THE SAME DIRECTION AS THE LINE BETWEEN THE TWO WRISTS
# WE GUESS THE BATS LENGTH BY MEASURING FROM THE MIDDLE OF THE HIPS TO THE FRONT TOE
def get_bat_line(left_wrist, right_wrist, left_hip, left_ankle, front_foot_side, hip_midpoint, front_toe):
    """
    DRAWS THE BAT LINE
    """
    if left_wrist is None or right_wrist is None or hip_midpoint is None or front_toe is None:
        return None, None
    
    # FIGURE OUT WHICH WRIST IS THE FRONT ONE
    if front_foot_side == "LEFT":
        front_wrist = left_wrist
    else:
        front_wrist = right_wrist
    
    direction_vector = right_wrist - left_wrist
    
    bat_length = np.linalg.norm(hip_midpoint - front_toe)
    
    # DO SOME MATH TO CREATE THE BAT VECTOR
    if np.linalg.norm(direction_vector) > 0:
        bat_vector = (direction_vector / np.linalg.norm(direction_vector)) * bat_length
        
        bat_start = front_wrist
        bat_end = front_wrist + bat_vector
        
        return tuple(bat_start.astype(int)), tuple(bat_end.astype(int))
    
    return None, None

# A LITTLE MATH HELPER GIVEN A LINE FROM P1 TO P2 AND A POINT P3
# IT FINDS THE SPOT ON THE LINE THAT IS CLOSEST TO THE POINT
# WE USE THIS TO FIGURE OUT WHERE ON OUR BATLINE THE BALL MADE CONTACT
def find_closest_point_on_line(p1, p2, p3):
    """
    FINDS THE POINT ON THE LINE SEGMENT P1P2 THAT IS CLOSEST TO P3
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    line_vec = p2 - p1
    p3_vec = p3 - p1
    
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq == 0:
        return p1
    
    t = np.dot(p3_vec, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)
    
    closest_point = p1 + t * line_vec
    return closest_point

# THIS IS THE MAIN FUNCTION THAT DOES ALL THE WORK
def video_dekho(video_path, output_folder="./output"):
    print(f"Reading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    
    # GET SOME BASIC INFO ABOUT THE VIDEO
    os.makedirs(output_folder, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {width}x{height} @ {fps} FPS, {frames} frames")
    
    count = 0
    
    # SET UP THE FILE WHERE WELL SAVE OUR NEW VIDEO WITH ALL THE DRAWINGS ON IT
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    output_path = os.path.join(output_folder, "annotated_video.mp4")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    start_time = time.time()
    
    # GET THE MEDIAPIPE TOOLS READY
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # WE NEED EMPTY LISTS TO STORE ALL THE MEASUREMENTS WE TAKE FROM EACH FRAME
    elbow_angles = []
    spine_leans = []
    head_over_knee_offsets = []
    foot_angles = []
    
    # THESE ARE OUR IDEAL NUMBERS FOR A GOOD CRICKET SHOT WELL COMPARE AGAINST THESE
    cfg = {
        "front_elbow_good_min": 100,
        "front_elbow_good_max": 150,
        "spine_lean_max_good": 20,
        "head_over_knee_max_px": 60,
        "front_foot_angle_max_good": 25,
    }
    
    # WE CAN BREAK A CRICKET SHOT DOWN INTO DIFFERENT PHASES
    class PHASE:
        STANCE = "Stance"
        STRIDE = "Stride"
        DOWNSWING = "Downswing"
        IMPACT = "Impact"
        FOLLOW_THROUGH = "FollowThrough"
        RECOVERY = "Recovery"
    
    current_phase = PHASE.STANCE
    phase_history = []
    
    prev_landmarks = None
    wrist_velocities = []
    
    # THIS IS THE MAIN LOOP IT GOES THROUGH THE VIDEO FRAME BY FRAME
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MEDIAPIPE LIKES RGB IMAGES BUT OPENCV READS THEM AS BGR SO WE HAVE TO CONVERT IT
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        landmarks = {}
        # IF MEDIAPIPE FOUND A PERSON IN THE FRAME
        if results.pose_landmarks:
            # GET THE COORDINATES FOR ALL THE BODY PARTS WE CARE ABOUT
            landmarks = {k: get_coordinates(results.pose_landmarks, width, height, v) for k, v in MARKS.items()}
            
            # CALCULATE MIDPOINTS FOR SHOULDERS AND HIPS THEY ARE USEFUL
            shoulder_midpoint = None
            if landmarks.get("LEFT_SHOULDER") is not None and landmarks.get("RIGHT_SHOULDER") is not None:
                shoulder_midpoint = (landmarks["LEFT_SHOULDER"] + landmarks["RIGHT_SHOULDER"]) / 2
            
            hip_midpoint = None
            if landmarks.get("LEFT_HIP") is not None and landmarks.get("RIGHT_HIP") is not None:
                hip_midpoint = (landmarks["LEFT_HIP"] + landmarks["RIGHT_HIP"]) / 2
            
            # FIGURE OUT WHICH SIDE IS THE FRONT
            front_foot_side = extract_front_foot(
                landmarks.get("LEFT_KNEE"),
                landmarks.get("RIGHT_KNEE"),
                landmarks.get("LEFT_HIP"),
                landmarks.get("RIGHT_HIP")
            )
            
            # GET ALL THE KEY LANDMARKS FOR THE FRONT SIDE OF THE BODY
            front_shoulder = landmarks.get(f"{front_foot_side}_SHOULDER")
            front_elbow = landmarks.get(f"{front_foot_side}_ELBOW")
            front_wrist = landmarks.get(f"{front_foot_side}_WRIST")
            front_knee = landmarks.get(f"{front_foot_side}_KNEE")
            front_toe = landmarks.get(f"{front_foot_side}_FOOT_INDEX")
            front_heel = landmarks.get(f"{front_foot_side}_HEEL")
            head = landmarks.get("NOSE")
            
            # NOW CALCULATE ALL OUR METRICS FOR THIS FRAME
            elbow = calc_angle(front_shoulder, front_elbow, front_wrist)
            if elbow is not None:
                elbow_angles.append(elbow)
            else:
                elbow_angles.append(np.nan)
            
            if shoulder_midpoint is not None and hip_midpoint is not None:
                # TO MEASURE SPINE LEAN WE CHECK THE ANGLE OF THE LINE FROM HIPS TO SHOULDERS AGAINST A PERFECTLY VERTICAL LINE
                spine = calc_angle(shoulder_midpoint, hip_midpoint, np.array([hip_midpoint[0], hip_midpoint[1] + 100], dtype=np.float32))
                if spine is not None:
                    spine_leans.append(spine)
                else:
                    spine_leans.append(np.nan)
            else:
                spine_leans.append(np.nan)

            if head is not None and front_knee is not None:
                head_offset = abs(head[0] - front_knee[0])
                head_over_knee_offsets.append(head_offset)
            else:
                head_over_knee_offsets.append(np.nan)
            
            foot_angle = ang_to_x(front_toe - front_heel) if front_toe is not None and front_heel is not None else None
            if foot_angle is not None:
                foot_angles.append(foot_angle)
            else:
                foot_angles.append(np.nan)

            # --- PHASE DETECTION LOGIC ---
            # WE FIGURE OUT THE PHASE OF THE SHOT BY LOOKING AT HOW FAST THE WRIST IS MOVING
            # LETS JUST TRACK THE RIGHT WRIST FOR SPEED
            current_wrist = landmarks.get("RIGHT_WRIST") 
            if prev_landmarks is not None and current_wrist is not None and prev_landmarks.get("RIGHT_WRIST") is not None:
                # VELOCITY IS JUST THE DISTANCE MOVED DIVIDED BY TIME TIME IS 1FPS
                velocity = np.linalg.norm(current_wrist - prev_landmarks.get("RIGHT_WRIST")) * fps
                wrist_velocities.append(velocity)
                
                # A SIMPLE STATE MACHINE TO CHANGE PHASES BASED ON SPEED
                if current_phase == PHASE.STANCE and velocity > 15:
                    current_phase = PHASE.STRIDE
                elif current_phase == PHASE.STRIDE and velocity > 50:
                    current_phase = PHASE.DOWNSWING
                # THE PEAK OF THE DOWNSWING IS THE IMPACT
                elif current_phase == PHASE.DOWNSWING and len(wrist_velocities) > 10 and velocity == np.max(wrist_velocities[-10:]):
                    current_phase = PHASE.IMPACT
                # AFTER IMPACT COMES THE FOLLOW THROUGH
                elif current_phase == PHASE.IMPACT:
                    current_phase = PHASE.FOLLOW_THROUGH
                # WHEN THINGS SLOW DOWN WERE IN RECOVERY
                elif current_phase == PHASE.FOLLOW_THROUGH and velocity < 5:
                    current_phase = PHASE.RECOVERY
            else:
                wrist_velocities.append(np.nan)

            prev_landmarks = landmarks
            
            # --- DRAWING OVERLAYS ON THE VIDEO FRAME ---
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )
        else:
            # IF NO PERSON WAS FOUND WE JUST ADD NAN NOT A NUMBER TO OUR DATA LISTS FOR THIS FRAME
            elbow_angles.append(np.nan)
            spine_leans.append(np.nan)
            head_over_knee_offsets.append(np.nan)
            foot_angles.append(np.nan)
            wrist_velocities.append(np.nan)
            prev_landmarks = None

        # TRY TO FIND THE BALL IN THE FRAME
        ball_pos = detect_ball(frame)
        
        # GET THE START AND END POINTS FOR OUR ESTIMATED BAT LINE
        bat_start, bat_end = get_bat_line(
            landmarks.get("LEFT_WRIST"), 
            landmarks.get("RIGHT_WRIST"), 
            landmarks.get("LEFT_HIP"), 
            landmarks.get("LEFT_ANKLE"),
            front_foot_side if 'front_foot_side' in locals() else "LEFT",  
            hip_midpoint if 'hip_midpoint' in locals() else None,
            front_toe if 'front_toe' in locals() else None 
        )

        # IF WE HAVE A BAT LINE DRAW IT
        if bat_start is not None and bat_end is not None:
            # A NICE CYAN COLOR FOR THE BAT
            cv2.line(frame, bat_start, bat_end, (255, 255, 0), 2) 
            
            # IF WE THINK ITS THE MOMENT OF IMPACT AND WE SEE A BALL AND A BAT
            if current_phase == PHASE.IMPACT and ball_pos is not None:
                # FIND THE EXACT POINT OF CONTACT
                impact_point = find_closest_point_on_line(bat_start, bat_end, ball_pos)
                if impact_point is not None:
                    # AND DRAW A BIG RED CIRCLE AND SOME TEXT THERE
                    cv2.circle(frame, tuple(impact_point.astype(int)), 10, (0, 0, 255), -1)
                    put_text(frame, "IMPACT!", tuple(impact_point.astype(int) - [0, 20]), scale=1.0, color=(0, 0, 255))
        
        # --- DISPLAY ALL THE LIVE FEEDBACK TEXT ON THE SCREEN ---
        put_text(frame, f"Phase: {current_phase}", (10, 300))
        
        elbow_live = get_last_valid(elbow_angles)
        spine_live = get_last_valid(spine_leans)
        hok_live = get_last_valid(head_over_knee_offsets)
        foot_live = get_last_valid(foot_angles)
        
        y_start = 30
        y_step = 24
        put_text(frame, f"Elbow: {elbow_live:.1f} deg" if elbow_live is not None else "Elbow: --", (10, y_start))
        put_text(frame, f"Spine lean: {spine_live:.1f} deg" if spine_live is not None else "Spine lean: --", (10, y_start+y_step))
        put_text(frame, f"Head over knee: {hok_live:.1f} px" if hok_live is not None else "Head over knee: --", (10, y_start+2*y_step))
        put_text(frame, f"Foot angle: {foot_live:.1f} deg" if foot_live is not None else "Foot angle: --", (10, y_start+3*y_step))
        
        # AND A SIMPLE CHECKLIST FOR GOOD TECHNIQUE
        yb = y_start + 5*y_step
        is_elbow_good = (elbow_live is not None) and (cfg["front_elbow_good_min"] <= elbow_live <= cfg["front_elbow_good_max"])
        is_spine_good = (spine_live is not None) and (spine_live <= cfg["spine_lean_max_good"])
        is_head_good = (hok_live is not None) and (hok_live <= cfg["head_over_knee_max_px"])
        is_foot_good = (foot_live is not None) and (foot_live <= cfg["front_foot_angle_max_good"])
        
        put_text(frame, f"{'V' if is_elbow_good else 'X'} Elbow elevation", (10, yb))
        put_text(frame, f"{'V' if is_spine_good else 'X'} Upright spine", (10, yb+y_step))
        put_text(frame, f"{'V' if is_head_good else 'X'} Head over front knee", (10, yb+2*y_step))
        put_text(frame, f"{'V' if is_foot_good else 'X'} Front foot direction", (10, yb+3*y_step))

        # WRITE THE FINAL FRAME WITH ALL OUR DRAWINGS TO THE OUTPUT VIDEO FILE
        video_writer.write(frame)
        count += 1
        
        # JUST A LITTLE PROGRESS UPDATE IN THE CONSOLE
        if count % 10 == 0:
            print(f"Processing frame {count}/{frames}...")

    # --- ALL DONE NOW WE CLEAN UP ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    pose.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if total_time > 0:
        processing_fps = count / total_time
        print(f"Average Processing FPS: {processing_fps:.2f}")

    print(f"Total Time taken = {total_time:.2f}s")
    print(f"Video analysis complete Processed {count} frames")

    # PACKAGE UP ALL OUR DATA
    metrics_data = {
        "elbow_angles": elbow_angles,
        "spine_leans": spine_leans,
        "head_over_knee_offsets": head_over_knee_offsets,
        "foot_angles": foot_angles,
        "wrist_velocities": wrist_velocities 
    }
    
    # AND CALL THE FUNCTIONS TO GENERATE THE FINAL REPORT AND THE GRAPHS
    final_evaluation(metrics_data, cfg, output_folder)
    generate_plots(metrics_data, output_folder) 

# THIS IS DONE FOR STREAMLIT 
if __name__ == "__main__":
    video_file = "./video.mp4"
    # CHECK IF THE VIDEO FILE ACTUALLY EXISTS BEFORE WE TRY TO OPEN IT
    if not os.path.exists(video_file):
        print("NO VIDEO FILE ")
    else:
        # IF IT EXISTS LETS GET TO WORK
        video_dekho(video_file)