import os
import time
import traceback
import requests
from datetime import datetime, timezone
from enum import Enum
import threading
import sys
import types
from PIL import Image, ImageDraw, ImageFont
import numpy as np

if sys.platform != 'darwin':
    import board
    import adafruit_ssd1306
    root_dir = "/home/pi/mbta/"
else:
    import cv2
    root_dir = "/Users/alexvahid/Documents/mbta_python/"



# ==========================
# CONFIG
# ==========================

MBTA_API_BASE = "https://api-v3.mbta.com"
MBTA_API_KEY = "ed7ec1d2391d4a4794873493d46ee345"

ROUTE_ID = "Green-D"
INBOUND_DIRECTION_ID = 1
START_STOP_ID = "place-river"
TARGET_STOP_ID = "place-bcnfd"
TARGET_STOP_SEQUENCE = 390

WIDTH = 128
HEIGHT = 64

API_REFRESH_SECONDS = 2
SCREEN_REFRESH_SECONDS = 0.01
SCROLL_SPEED = 3
LOG_DIR = f"{root_dir}mbta_93_logs"

MAP_WIDTH, MAP_HEIGHT = 1632, 835
img_map = Image.open(f"{root_dir}map_gl.png").convert("RGBA")
# zoom_factor=10.0
# MAP_WIDTH = int(ORIG_W * zoom_factor)
# MAP_HEIGHT = int(ORIG_H * zoom_factor)
# img_map = img_map.resize((MAP_WIDTH, MAP_HEIGHT), Image.LANCZOS)
EMPTY_IMAGE = Image.new("1", (WIDTH, HEIGHT))

# Load icon and resize
ICON_SIZE = int(18)
icon_src = Image.open(f"{root_dir}mbta.png").convert("RGBA").resize((ICON_SIZE, ICON_SIZE))
icon_composite = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 255))
icon_composite.alpha_composite(icon_src)

# Create the circular clipping mask
mask = Image.new("L", (ICON_SIZE, ICON_SIZE), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0, ICON_SIZE - 1, ICON_SIZE - 1), fill=255)

# ==========================
# SETUP
# ==========================

os.makedirs(LOG_DIR, exist_ok=True)
PROGRAM_START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"{LOG_DIR}/{PROGRAM_START_TIME}.txt"

HEADERS = {"x-api-key": MBTA_API_KEY}

# ==========================
# ENUM
# ==========================

class BusState(Enum):
    ON_ROUTE_TO_TARGET_STOP = 1
    AT_START_STATION = 2
    BEYOND_TARGET_STOP = 3

# ==========================
# API FUNCTIONS
# ==========================

def safe_get(url, params):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

def get_predictions():
    return safe_get(
        f"{MBTA_API_BASE}/predictions",
        {
            "filter[route]": ROUTE_ID,
            "filter[stop]": TARGET_STOP_ID,
            "filter[direction_id]": INBOUND_DIRECTION_ID,
            "sort": "arrival_time",
        },
    )

def get_vehicles():
    return safe_get(
        f"{MBTA_API_BASE}/vehicles",
        {
            "filter[route]": ROUTE_ID,
            "filter[direction_id]": INBOUND_DIRECTION_ID,
        },
    )


# ==========================
# DISPLAY SETUP
# ==========================

window_name = "MBTA"
def init_display():
    if sys.platform != 'darwin':
        i2c = board.I2C()
        oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c)
        oled.fill(0)
        oled.show()
        return oled, i2c
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1000, 500)
        return None, None

def display_image(oled, image):
    if sys.platform != 'darwin':
        oled.image(image)
        oled.show()
    else:

        opencv_image = np.array(image.convert('L'))
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imshow(window_name, opencv_image)
        cv2.waitKey(1) # Waits forever until a key is pressed




    # 4. Perspective Logic (Relative to the 500x500 crop)
    #Effect Desired	Pinch Adjustment	V-Squash Adjustment	Result
    #Current (Dramatic)	/ 2	* 0.5	Sharp "V" shape, cinematic horizon.
    #Realistic Map	/ 3	* 0.3	Wider horizon, feels like a GPS at a 45° angle.
    #Extreme "Star Wars" Text	/ 2.1	* 0.8	Extremely flat, text seems to disappear into the distance.
    #Subtle Tilt	/ 5	* 0.2	Very mild perspective, mostly keeps the rectangular shape.
def plot_on_map(target_lat, target_lon):
    # 1. Configuration
    TILT_AMOUNT = 0.96
    ZOOM_LEVEL = 10  # Desired zoom factor
    
    TOP_LEFT_LAT, TOP_LEFT_LON = 42.367667, -71.262748
    BOTTOM_RIGHT_LAT, BOTTOM_RIGHT_LON = 42.314685, -71.122894
    
    CROP_W, CROP_H = 128, 64
    PRE_TILT_CROP_W, PRE_TILT_CROP_H = 2000, 2000

    # 2. Calculate coordinates on the ORIGINAL (large) map
    lon_pct = (target_lon - TOP_LEFT_LON) / (BOTTOM_RIGHT_LON - TOP_LEFT_LON)
    lat_pct = (target_lat - TOP_LEFT_LAT) / (BOTTOM_RIGHT_LAT - TOP_LEFT_LAT)
    orig_px_x = lon_pct * MAP_WIDTH
    orig_px_y = lat_pct * MAP_HEIGHT

    # 3. MEMORY FIX: Crop a small patch from the original map first
    # Calculate how many original pixels we need to get 2000x2000 after 10x zoom
    buffer_w = PRE_TILT_CROP_W / ZOOM_LEVEL 
    buffer_h = PRE_TILT_CROP_H / ZOOM_LEVEL

    left = orig_px_x - (buffer_w / 2)
    top = orig_px_y - (buffer_h / 2)
    
    # Crop first (This is very memory efficient)
    small_patch = img_map.crop((left, top, left + buffer_w, top + buffer_h))

    # 4. Zoom only the small patch to the 2000x2000 canvas
    zoomed_crop = small_patch.resize((PRE_TILT_CROP_W, PRE_TILT_CROP_H), Image.LANCZOS)

    # 5. Local coordinates are now centered on the zoomed_crop
    local_px_x = PRE_TILT_CROP_W / 2
    local_px_y = PRE_TILT_CROP_H / 2

    # 6. Perspective Logic
    pinch = (PRE_TILT_CROP_W / 2.1) * TILT_AMOUNT
    v_squash = PRE_TILT_CROP_H * (TILT_AMOUNT * 0.8) 
    
    src_pts = np.array([(0, 0), (PRE_TILT_CROP_W, 0), (PRE_TILT_CROP_W, PRE_TILT_CROP_H), (0, PRE_TILT_CROP_H)])
    dst_pts = np.array([
        (pinch, v_squash),               
        (PRE_TILT_CROP_W - pinch, v_squash),   
        (PRE_TILT_CROP_W, PRE_TILT_CROP_H),         
        (0, PRE_TILT_CROP_H)                  
    ])
    
    forward_coeffs = find_coeffs(src_pts, dst_pts)
    render_coeffs = find_coeffs(dst_pts, src_pts) 

    tilted_map = zoomed_crop.transform((PRE_TILT_CROP_W, PRE_TILT_CROP_H), Image.PERSPECTIVE, render_coeffs, Image.BILINEAR)
    
    # 7. Project Point onto Tilted Surface
    a, b, c, d, e, f, g, h = forward_coeffs
    denom = (g * local_px_x + h * local_px_y + 1)
    new_px_x = (a * local_px_x + b * local_px_y + c) / denom
    new_px_y = (d * local_px_x + e * local_px_y + f) / denom

    # 8. Final Crop for Display
    final_left = max(0, min(PRE_TILT_CROP_W - CROP_W, int(new_px_x - (CROP_W / 2))))
    final_top = max(0, min(PRE_TILT_CROP_H - CROP_H, int(new_px_y - (CROP_H * 0.75))))
    
    final_image = tilted_map.crop((final_left, final_top, final_left + CROP_W, final_top + CROP_H))

    # 9. Paste Icon
    icon_x = int(new_px_x - final_left - (ICON_SIZE / 2))
    icon_y = int(new_px_y - final_top - (ICON_SIZE / 2))
    final_image.paste(icon_composite, (icon_x, icon_y), mask=mask)
    
    return final_image.convert("1", dither=Image.FLOYDSTEINBERG)

def should_show_map(state): 
    if state == BusState.ON_ROUTE_TO_TARGET_STOP or state == BusState.AT_START_STATION:
        return True
    else:
        return False

vehicles = []
predictions = []
map = EMPTY_IMAGE
def api_data_refresher():
    global vehicles
    global predictions
    global map
    while True:
        try:
            pred_data = get_predictions()
            veh_data = get_vehicles()

            temp_predictions = []
            if pred_data:
                for item in pred_data.get("data", []):
                    arrival = item["attributes"].get("arrival_time")
                    if arrival:
                        temp_predictions.append(
                            datetime.fromisoformat(arrival.replace("Z", f"+00:00"))
                        )
                    
                temp_predictions.sort()
                predictions = temp_predictions

            temp_vehicles = veh_data.get("data", []) if veh_data else []
            if temp_vehicles:
                for v in temp_vehicles:
                    attrs = v.get("attributes", {})
                    rel = v.get("relationships", {}).get("stop", {}).get("data")

                    stop_seq = attrs.get("current_stop_sequence") or 0
                    stop_id = rel["id"] if rel else None

                    if stop_seq >= TARGET_STOP_SEQUENCE:
                        v["state"] = BusState.BEYOND_TARGET_STOP
                    elif stop_id != START_STOP_ID:
                        v["state"] = BusState.ON_ROUTE_TO_TARGET_STOP
                    else:
                        v["state"] = BusState.AT_START_STATION

                temp_vehicles.sort(key=lambda x: x["state"].value)
                vehicles = temp_vehicles

            if vehicles:
                approaching_vehicle = vehicles[0]
                if should_show_map(approaching_vehicle["state"]):
                    approaching_vehicle_loc = [approaching_vehicle["attributes"]["latitude"],approaching_vehicle["attributes"]["longitude"]]
                    map = plot_on_map(approaching_vehicle_loc[0], approaching_vehicle_loc[1])
                else:
                    map = EMPTY_IMAGE
            else:
                map = EMPTY_IMAGE

        except Exception as e:
            print(f"Background Fetch Error: {e}")

        time.sleep(API_REFRESH_SECONDS)

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return np.array(res).reshape(8)

# ==========================
# MAIN
# ==========================

def main():
    global vehicles
    global predictions
    global map
    oled, i2c = init_display()

    t_image = Image.open(f"{root_dir}mbta.png")

    # Resize image to fit the display dimensions
    # Use Image.BICUBIC for a decent resizing quality
    t_image.thumbnail((WIDTH, HEIGHT), Image.BICUBIC)

    # Convert the image to 1-bit color mode ('1') required by the SSD1306
    # This step is crucial for monochrome displays
    t_image = t_image.convert('1')

    # # Ensure the final image is exactly the right size if the thumbnail function didn't make it exact
    if t_image.width != WIDTH or t_image.height != HEIGHT:
        # Create a new blank image with the target dimensions
        # color=0 creates a black background; use color=1 or 255 for white depending on mode
        final_image = Image.new("1", (WIDTH, HEIGHT), color=0)
        
        # Calculate centering coordinates
        upper_left_x = (WIDTH - t_image.width) // 2
        upper_left_y = (HEIGHT - t_image.height) // 2
        
        # Paste the thumbnail into the center
        final_image.paste(t_image, (upper_left_x, upper_left_y))
        t_image = final_image

    image = Image.new("1", (WIDTH, HEIGHT))
    main_draw = ImageDraw.Draw(image)

    prediction_font = ImageFont.truetype(
        f"{root_dir}HelveticaNeue-Medium.otf", 45
    )
    message_font = ImageFont.truetype(
        f"{root_dir}HelveticaNeue-Medium.otf", 17
    )

    prediction_font_mini = ImageFont.truetype(
        f"{root_dir}HelveticaNeue-Medium.otf", 20
    )
    message_font_mini = ImageFont.truetype(
        f"{root_dir}HelveticaNeue-Medium.otf", 13
    )

    show_colon = True
    last_api_fetch = 0
    last_colon_toggle = 0 
    predictions = []
    vehicles = []

    last_log_time = 0
    scroll_x = WIDTH 

    api_thread = threading.Thread(target=api_data_refresher, daemon=True)
    api_thread.start()

    while True:
        try:
            now_ts = time.time()

            # ==========================
            # CALCULATE TEXT
            # ==========================
            prediction_text = "--:--"
            approaching_vehicle_status = "NO BUS"
            approaching_vehicle_state = None

            if predictions:
                now = datetime.now(timezone.utc)
                seconds = max(0, int((predictions[0] - now).total_seconds()))
                prediction_text = f"{seconds//60}{":" if show_colon else " "}{seconds%60:02d}"

            if vehicles:
                approaching_vehicle = vehicles[0]
                approaching_vehicle_state = approaching_vehicle["state"]

                if approaching_vehicle_state == BusState.ON_ROUTE_TO_TARGET_STOP:
                    approaching_vehicle_status = f"+{TARGET_STOP_SEQUENCE - approaching_vehicle["attributes"]["current_stop_sequence"]} STOPS"
                elif approaching_vehicle_state == BusState.AT_START_STATION:
                    approaching_vehicle_status = "AT SULLIVAN"

                

            # ==========================
            # DRAW
            # ==========================
            if now_ts - last_colon_toggle > 1.0:
                show_colon = not show_colon
                last_colon_toggle = now_ts

            if not should_show_map(approaching_vehicle_state):
                main_draw.rectangle((0, 0, WIDTH, HEIGHT), fill=0)

                pw, ph = prediction_font.getbbox(prediction_text)[2:]
                main_draw.text(
                    ((WIDTH - pw) // 2, 2),
                    prediction_text,
                    font=prediction_font,
                    fill=255,
                )

                mw, mh = message_font.getbbox(approaching_vehicle_status)[2:]
                main_draw.text(
                    (scroll_x, HEIGHT - mh - 2),
                    approaching_vehicle_status,
                    font=message_font,
                    fill=255,
                )

                scroll_x -= SCROLL_SPEED
                if scroll_x < -mw:
                    scroll_x = WIDTH

                screen_image = image
            else:
                map_copy = map.copy()
                map_draw = ImageDraw.Draw(map_copy)

                pw, ph = prediction_font_mini.getbbox(prediction_text)[2:]
                map_draw.rectangle((0, 0, pw, ph), fill=0)

                map_draw.text(
                    (0, 0),
                    prediction_text,
                    font=prediction_font_mini,
                    fill=255,
                )

                mw, mh = message_font_mini.getbbox(approaching_vehicle_status)[2:]
                map_draw.rectangle((WIDTH - mw, 0, WIDTH, mh), fill=0)
                map_draw.text(
                    (WIDTH - mw, 0),
                    approaching_vehicle_status,
                    font=message_font_mini,
                    fill=255,
                )

                screen_image = map_copy

            # ==========================
            # OLED UPDATE (WITH RECOVERY)
            # ==========================
            try:
                display_image(oled, screen_image)

            except OSError:
                print("screen err")
                i2c.deinit()
                oled, i2c = init_display()

        except Exception as e:
            print(e)
            if time.time() - last_log_time > 60:
                with open(LOG_FILE, "a") as f:
                    traceback.print_exc(file=f)
                last_log_time = time.time()

        time.sleep(SCREEN_REFRESH_SECONDS)
        

# ==========================
# ENTRY
# ==========================

if __name__ == "__main__":
    main()
