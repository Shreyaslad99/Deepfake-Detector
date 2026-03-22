import cv2
from PIL import Image
from .models import predict_image

def predict_video(video_path, num_frames=15):
    """
    Slices a video into evenly spaced frames and analyzes them.
    Uses 'Top-K' scoring to ensure highly suspicious glitch frames 
    aren't diluted by perfectly rendered frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Could not read video file or video is empty.")
        
    # Calculate the step to grab 'num_frames' evenly across the video length
    step = max(total_frames // num_frames, 1)
    
    frame_scores = []
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert OpenCV's BGR format to standard RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Analyze this specific frame
        score = predict_image(pil_img)
        frame_scores.append(score)
        
    cap.release()
    
    if not frame_scores:
        raise ValueError("Failed to extract any frames from the video.")
        
    # Sort scores from highest (most fake) to lowest (most real)
    frame_scores.sort(reverse=True)

    # 1. Peak Score (The single most obvious AI frame)
    max_score = frame_scores[0]
    
    # 2. Overall Average
    avg_score = sum(frame_scores) / len(frame_scores)

    # 3. 🔥 TOP-K AVERAGE (The Game Changer)
    # Take the top 30% most suspicious frames (minimum of 3 frames).
    # This zeroes in on the specific moments the deepfake generation struggled.
    k = max(3, len(frame_scores) // 3)
    top_k_avg = sum(frame_scores[:k]) / k

    # NEW WEIGHTING FORMULA:
    # 50% Top-K Average: Heavily targets the glitchy/suspicious moments.
    # 30% Peak Score: Ensures a massive single-frame failure spikes the score.
    # 20% Overall Average: Keeps the baseline context of the whole video.
    final_video_score = (top_k_avg * 0.5) + (max_score * 0.3) + (avg_score * 0.2)
    
    # --- DEBUGGING TERMINAL OUTPUT ---
    print("\n" + "="*40)
    print(f"[DEBUG] Video Frames Analyzed: {len(frame_scores)}")
    print(f"[DEBUG] Peak AI Frame Score: {round(max_score * 100, 2)}%")
    print(f"[DEBUG] Top-{k} Suspicious Frames Avg: {round(top_k_avg * 100, 2)}%")
    print(f"[DEBUG] Full Video Average: {round(avg_score * 100, 2)}%")
    print(f"[DEBUG] FINAL WEIGHTED SCORE: {round(final_video_score * 100, 2)}%")
    print("="*40 + "\n")

    return final_video_score, max_score