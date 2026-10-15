import os
import cv2

def test_opencv_setup(image_path="data/input/sample.jpg", video_path="data/input/sample.mp4"):
    print("=" * 50)
    print("OpenCV Environment & Media Test")
    print("=" * 50)
    
    # 1. OpenCV Version Check
    print(f"[+] OpenCV Version: {cv2.__version__}")
    
    # 2. Image Test
    print("\n--- Image Verification ---")
    if not os.path.exists(image_path):
        print(f"[-] Image file NOT found at: {image_path}")
        print(f"    Please place a sample image file at: {os.path.abspath(image_path)}")
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[-] Failed to load image at: {image_path} (Invalid image format or corrupt file)")
        else:
            height, width, channels = img.shape
            print(f"[+] Image loaded successfully from: {image_path}")
            print(f"    Dimensions: {width}x{height} (Width x Height), Channels: {channels}")

    # 3. Video Test
    print("\n--- Video Verification ---")
    if not os.path.exists(video_path):
        print(f"[-] Video file NOT found at: {video_path}")
        print(f"    Please place a sample video file at: {os.path.abspath(video_path)}")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[-] Failed to open video at: {video_path}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[+] Video opened successfully from: {video_path}")
            print(f"    Resolution : {width}x{height}")
            print(f"    FPS        : {fps:.2f}")
            print(f"    Total Frames: {frame_count}")
            cap.release()
            
    print("=" * 50)

if __name__ == "__main__":
    test_opencv_setup()
