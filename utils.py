from enum import Enum
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import requests
from datetime import datetime, timezone
import sys
import copy
import traceback
import time
import os
import re

if sys.platform != 'darwin':
    import board
    import adafruit_ssd1306
    ROOT_DIR = "/home/pi/mbta/"
else:
    import cv2
    ROOT_DIR = "/Users/alexvahid/Documents/mbta_python/"

LOG_DIR = f"{ROOT_DIR}mbta_logs"
MBTA_API_BASE = "https://api-v3.mbta.com"
MBTA_API_KEY = "ed7ec1d2391d4a4794873493d46ee345"
HEADERS = {"x-api-key": MBTA_API_KEY}
WIDTH = 128
HEIGHT = 64
WINDOW_NAME = "MBTA"
EMPTY_IMAGE = Image.new("1", (WIDTH, HEIGHT))
VEHICLE_REFRESH_SECONDS = 2
MAP_REFRESH_SECONDS = 3
PREDICTION_REFRESH_SECONDS = 2
ICON_SIZE = int(24)
ICON_COMPOSITE = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 255))
ICON_COMPOSITE.alpha_composite(Image.open(f"{ROOT_DIR}mbta.png").convert("RGBA").resize((ICON_SIZE, ICON_SIZE)))
ICON_MASK = Image.new("L", (ICON_SIZE, ICON_SIZE), 0)
draw = ImageDraw.Draw(ICON_MASK)
draw.ellipse((0, 0, ICON_SIZE - 1, ICON_SIZE - 1), fill=255)


vehicles = []
predictions = []
background = None
map = EMPTY_IMAGE
debug_index = 0

os.makedirs(LOG_DIR, exist_ok=True)
PROGRAM_START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"{LOG_DIR}/{PROGRAM_START_TIME}.txt"

class BusState(Enum):
    ON_ROUTE_TO_TARGET_STOP = 1
    AT_START_STATION = 2
    BEYOND_TARGET_STOP = 3

def get_font(font_name, size):
    return ImageFont.truetype(f"{ROOT_DIR}{font_name}", size)
BRAT_FONT = get_font("arialnarrow.ttf", 14)

def init_display():
    if sys.platform != 'darwin':
        i2c = board.I2C()
        oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c)
        oled.fill(0)
        oled.show()
        return oled, i2c
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW_NAME, 1000, 500)
        return None, None

def display_image(oled, image):
    if sys.platform != 'darwin':
        oled.image(image)
        oled.show()
    else:
        opencv_image = np.array(image.convert('L'))
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imshow(WINDOW_NAME, opencv_image)
        cv2.waitKey(1) 

def safe_get(url, params):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

def get_predictions(route_id, target_stop_id, direction_id):
    pred_data = safe_get(
        f"{MBTA_API_BASE}/predictions",
        {
            "filter[route]": route_id,
            "filter[stop]": target_stop_id,
            "filter[direction_id]": direction_id,
            "sort": "arrival_time",
        },
    )
    temp_predictions = []
    if pred_data:
        for item in pred_data.get("data", []):
            arrival = item["attributes"].get("arrival_time")
            if arrival:
                temp_predictions.append(
                    datetime.fromisoformat(arrival.replace("Z", f"+00:00"))
                )
            
        temp_predictions.sort()
    return temp_predictions

def get_vehicles(route_id, direction_id, start_sequence, target_stop_sequence):
    veh_data = safe_get(
        f"{MBTA_API_BASE}/vehicles",
        {
            "filter[route]": route_id,
            "filter[direction_id]": direction_id,
        },
    )
    temp_vehicles = veh_data.get("data", []) if veh_data else []
    if temp_vehicles:
        for v in temp_vehicles:
            attrs = v.get("attributes", {})
            rel = v.get("relationships", {}).get("stop", {}).get("data")

            stop_seq = attrs.get("current_stop_sequence") or 0
            stop_id = rel["id"] if rel else None

            if stop_seq >= target_stop_sequence:
                v["state"] = BusState.BEYOND_TARGET_STOP
            elif stop_id != start_sequence:
                v["state"] = BusState.ON_ROUTE_TO_TARGET_STOP
            else:
                v["state"] = BusState.AT_START_STATION

        temp_vehicles.sort(key=lambda x: x["state"].value)
    return temp_vehicles

class Perspective(Enum):
    DRAMATIC = (2, 0.5)
    GPS = (3, 0.3)
    STAR_WARS = (2.1, 0.8)
    SUBTLE_TILT = (5, 0.2)

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return np.array(res).reshape(8)



def calculate_target_pixels(
    base,
    target_pos
):
    # Unpack context
    tl_lat, tl_lon = base["top_left_pos"]
    br_lat, br_lon = base["bottom_right_pos"]
    map_w, map_h = base["map_size"]

    center_orig_x, center_orig_y = base["center_orig_px"]
    zoom = base["zoom"]

    pre_w, pre_h = base["pre_tilt_size"]
    final_left = base["final_left"]
    final_top = base["final_top"]

    target_lat, target_lon = target_pos

    # --- Convert target lat/lon to original map pixels ---
    lon_pct = (target_lon - tl_lon) / (br_lon - tl_lon)
    lat_pct = (target_lat - tl_lat) / (br_lat - tl_lat)

    orig_px_x = lon_pct * map_w
    orig_px_y = lat_pct * map_h

    # --- Offset from center (in original map pixels) ---
    dx = (orig_px_x - center_orig_x) * zoom
    dy = (orig_px_y - center_orig_y) * zoom

    # --- Pre-tilt local coordinates ---
    local_x = pre_w / 2 + dx
    local_y = pre_h / 2 + dy

    # --- Perspective projection ---
    a, b, c, d, e, f, g, h = base["forward_coeffs"]
    denom = g * local_x + h * local_y + 1

    proj_x = (a * local_x + b * local_y + c) / denom
    proj_y = (d * local_x + e * local_y + f) / denom

    return proj_x, proj_y

def plot_icon_on_map(
    base,
    icon_img,
    icon_mask,
    target_pos
):
    proj_x, proj_y = calculate_target_pixels(base, target_pos)

    # --- Paste icon ---
    icon_x = int(proj_x - base["final_left"] - icon_img.width / 2)
    icon_y = int(proj_y - base["final_top"] - icon_img.height / 2)

    base['base_image'].paste(icon_img, (icon_x, icon_y), mask=icon_mask)
    return base['base_image']

def plot_text_on_map(
    base,
    text,
    target_pos
):
    map_draw = ImageDraw.Draw(base['base_image'])
    proj_x, proj_y = calculate_target_pixels(base, target_pos)

    w, h = BRAT_FONT.getbbox(text)[2:]

    text_x = int(proj_x - base["final_left"] - w / 2)
    text_y = int(proj_y - base["final_top"] - h / 2)

    map_draw.text(
        (text_x, text_y),
        text,
        font=BRAT_FONT,
        stroke_width=3,
        stroke_fill=0,
        fill=255,
    )


def render_map_view(
    center_target_pos,
    img_map,
    top_left_pos,
    bottom_right_pos,
    perspective,
    zoom,
    landmarks
):
    TILT_AMOUNT = 0.96
    CROP_W, CROP_H = 128, 64
    PRE_TILT_CROP_W, PRE_TILT_CROP_H = 2000, 2000

    map_width, map_height = img_map.size
    tl_lat, tl_lon = top_left_pos
    br_lat, br_lon = bottom_right_pos
    center_lat, center_lon = center_target_pos

    # --- Original map coordinates ---
    lon_pct = (center_lon - tl_lon) / (br_lon - tl_lon)
    lat_pct = (center_lat - tl_lat) / (br_lat - tl_lat)

    orig_px_x = lon_pct * map_width
    orig_px_y = lat_pct * map_height

    # --- Crop ---
    buffer_w = PRE_TILT_CROP_W / zoom
    buffer_h = PRE_TILT_CROP_H / zoom

    left = orig_px_x - buffer_w / 2
    top = orig_px_y - buffer_h / 2

    small_patch = img_map.crop((left, top, left + buffer_w, top + buffer_h))
    zoomed_crop = small_patch.resize(
        (PRE_TILT_CROP_W, PRE_TILT_CROP_H), Image.LANCZOS
    )

    # --- Perspective ---
    pinch = (PRE_TILT_CROP_W / perspective.value[0]) * TILT_AMOUNT
    v_squash = PRE_TILT_CROP_H * (TILT_AMOUNT * perspective.value[1])

    src_pts = np.array([
        (0, 0),
        (PRE_TILT_CROP_W, 0),
        (PRE_TILT_CROP_W, PRE_TILT_CROP_H),
        (0, PRE_TILT_CROP_H)
    ])

    dst_pts = np.array([
        (pinch, v_squash),
        (PRE_TILT_CROP_W - pinch, v_squash),
        (PRE_TILT_CROP_W, PRE_TILT_CROP_H),
        (0, PRE_TILT_CROP_H)
    ])

    forward_coeffs = find_coeffs(src_pts, dst_pts)
    render_coeffs = find_coeffs(dst_pts, src_pts)

    tilted_map = zoomed_crop.transform(
        (PRE_TILT_CROP_W, PRE_TILT_CROP_H),
        Image.PERSPECTIVE,
        render_coeffs,
        Image.BILINEAR
    )

    # --- Project center point ---
    cx = PRE_TILT_CROP_W / 2
    cy = PRE_TILT_CROP_H / 2
    a, b, c, d, e, f, g, h = forward_coeffs
    denom = g * cx + h * cy + 1

    proj_x = (a * cx + b * cy + c) / denom
    proj_y = (d * cx + e * cy + f) / denom

    final_left = int(proj_x - CROP_W / 2)
    final_top = int(proj_y - CROP_H * 0.75)

    final_image = tilted_map.crop(
        (final_left, final_top, final_left + CROP_W, final_top + CROP_H)
    )

    background = {
        "forward_coeffs": forward_coeffs,
        "final_left": final_left,
        "final_top": final_top,
        "crop_size": (CROP_W, CROP_H),
        "pre_tilt_size": (PRE_TILT_CROP_W, PRE_TILT_CROP_H),
        "top_left_pos": top_left_pos,
        "bottom_right_pos": bottom_right_pos,
        "map_size": (map_width, map_height),
        "center_orig_px": (orig_px_x, orig_px_y),
        "zoom": zoom,
        "base_image": final_image.convert("1", dither=Image.FLOYDSTEINBERG)
    }

    for pos, text in landmarks:
        plot_text_on_map(background, text, pos)

    return background

def get_target_pos(vehicles, debug, debug_coordinates): 
    global debug_index
    if debug:
        return debug_coordinates[debug_index % 3]
    elif vehicles and (vehicles[0]["state"] == BusState.ON_ROUTE_TO_TARGET_STOP or vehicles[0]["state"] == BusState.AT_START_STATION):
        return vehicles[0]["attributes"]["latitude"], vehicles[0]["attributes"]["longitude"]
    else:
        return None

def background_refresher(img_map, top_left_pos, bottom_right_pos, landmarks, debug, debug_coordinates):
    global background
    global vehicles

    while True:
        try:
            target_pos = get_target_pos(copy.deepcopy(vehicles), debug, debug_coordinates)
            if target_pos:
                background = render_map_view(target_pos, img_map, top_left_pos, bottom_right_pos, Perspective.STAR_WARS, 5, landmarks)

        except Exception as e:
            print(f"API thread fetch error: {e}")
            traceback.print_exc()

        time.sleep(MAP_REFRESH_SECONDS)

def vehicle_data_refresher(route_id, direction_id, target_stop_id, target_stop_sequence, empty_image, debug, debug_coordinates):
    global vehicles
    global map
    global background
    global debug_index
    
    api_refresh_seconds = VEHICLE_REFRESH_SECONDS
    while True:
        try:
            vehicles_local_var = get_vehicles(route_id, direction_id, target_stop_id, target_stop_sequence)
            if debug:
                debug_index = debug_index + 1
            target_pos = get_target_pos(vehicles_local_var, debug, debug_coordinates)
            vehicles = vehicles_local_var

            if target_pos and background:
                map = plot_icon_on_map(copy.deepcopy(background), ICON_COMPOSITE, ICON_MASK, target_pos)
                api_refresh_seconds = 0
            else:
                map = empty_image
                api_refresh_seconds = VEHICLE_REFRESH_SECONDS
                

        except Exception as e:
            print(f"Vehicle thread fetch error: {e}")
            traceback.print_exc()

        time.sleep(api_refresh_seconds)

def prediction_data_refresher(route_id, target_stop_id, direction_id):
    global predictions
    
    while True:
        try:
            predictions = get_predictions(route_id, target_stop_id, direction_id)    

        except Exception as e:
            print(f"Prediction thread fetch error: {e}")
            traceback.print_exc()

        time.sleep(PREDICTION_REFRESH_SECONDS)

def main(target_stop_sequence, screen_refresh_seconds, scroll_speed, debug, debug_coordinates):
    global vehicles
    global predictions
    global map

    oled, i2c = init_display()

    image = Image.new("1", (WIDTH, HEIGHT))
    main_draw = ImageDraw.Draw(image)

    prediction_font_large  = get_font("HelveticaNeue-Medium.otf", 45)
    prediction_font_medium = get_font("HelveticaNeue-Medium.otf", 35)
    # prediction_font_mini   = get_font("HelveticaNeue-Medium.otf", 20)

    message_font_large = get_font("HelveticaNeue-Medium.otf", 17)
    # message_font_medium = get_font("HelveticaNeue-Medium.otf", 15)
    # message_font_mini = get_font("HelveticaNeue-Medium.otf", 11)

    frowny_font = get_font("FreeMono.ttf", 22)
    brat_font = get_font("arialnarrow.ttf", 17)

    show_colon = True
    last_colon_toggle = 0 

    last_log_time = 0
    scroll_x = WIDTH 


    while True:
        try:
            now_ts = time.time()

            prediction_text = "--:--"
            approaching_vehicle_status = "NO BUS "
            approaching_vehicle_state = None

            if predictions:
                now = datetime.now(timezone.utc)
                seconds = max(0, int((predictions[0] - now).total_seconds()))
                prediction_text = f"{seconds//60}{":" if show_colon else " "}{seconds%60:02d}"

            if now_ts - last_colon_toggle > 1.0:
                show_colon = not show_colon
                last_colon_toggle = now_ts
            
            vehicles_local_var = copy.deepcopy(vehicles)
            if get_target_pos(vehicles_local_var, debug, debug_coordinates):
                if vehicles_local_var:
                    approaching_vehicle = vehicles_local_var[0]
                    approaching_vehicle_state = approaching_vehicle["state"]

                    if approaching_vehicle_state == BusState.ON_ROUTE_TO_TARGET_STOP:
                        approaching_vehicle_status = f"{target_stop_sequence - approaching_vehicle["attributes"]["current_stop_sequence"]}"
                    elif approaching_vehicle_state == BusState.AT_START_STATION:
                        approaching_vehicle_status = ""
                if debug:
                    approaching_vehicle_status = "2"

                map_copy = map.copy()
                map_draw = ImageDraw.Draw(map_copy)
                prediction_minutes, prediction_seconds = re.search(r"([\d-]{1,2})([:\s][\d-]{2})", prediction_text).groups()
                
                pmw, pmh = prediction_font_medium.getbbox(prediction_minutes)[2:]
                map_draw.rectangle((0, 0, pmw, pmh + 1), fill=0)
                map_draw.text(
                    (0, 0),
                    prediction_minutes,
                    font=prediction_font_medium,
                    fill=255,
                )

                psw, psh = brat_font.getbbox(prediction_seconds)[2:]
                map_draw.rectangle((pmw, 0, pmw + psw, psh), fill=0)
                map_draw.text(
                    (pmw, -4),
                    prediction_seconds,
                    font=brat_font,
                    fill=255,
                )

                if approaching_vehicle_status:
                    mw, mh = prediction_font_medium.getbbox(approaching_vehicle_status)[2:]
                    map_draw.rectangle((WIDTH - mw, 0, WIDTH, mh), fill=0)
                    map_draw.text(
                        (WIDTH - mw, 0),
                        approaching_vehicle_status,
                        font=prediction_font_medium,
                        fill=255,
                    )

                    plus = "stops"
                    plus_w, plus_h = brat_font.getbbox(plus)[2:]
                    map_draw.rectangle((WIDTH - mw - plus_w, 0, WIDTH - mw, plus_h), fill=0)
                    map_draw.text(
                        (WIDTH - mw - plus_w, -6),
                        plus,
                        font=brat_font,
                        fill=255,
                    )

                screen_image = map_copy
            
            else:
                main_draw.rectangle((0, 0, WIDTH, HEIGHT), fill=0)

                pw, ph = prediction_font_large.getbbox(prediction_text)[2:]
                main_draw.text(
                    ((WIDTH - pw) // 2, 2),
                    prediction_text,
                    font=prediction_font_large,
                    fill=255,
                )

                mw, mh = message_font_large.getbbox(approaching_vehicle_status)[2:]
                main_draw.text(
                    (scroll_x, HEIGHT - mh - 2),
                    approaching_vehicle_status,
                    font=message_font_large,
                    fill=255,
                )

                frowny = "\u2639"
                frowny_w, frowny_h = frowny_font.getbbox(frowny)[2:]
                main_draw.text(
                    (scroll_x + mw + 1, HEIGHT - frowny_h - 2),
                    frowny,
                    font=frowny_font,
                    fill=255,
                )

                scroll_x -= scroll_speed
                if scroll_x < -(mw + frowny_w):
                    scroll_x = WIDTH

                screen_image = image

            try:
                display_image(oled, screen_image)

            except OSError:
                print("screen err")
                i2c.deinit()
                oled, i2c = init_display()

        except Exception as e:
            print(e)
            traceback.print_exc()
            if time.time() - last_log_time > 60:
                with open(LOG_FILE, "a") as f:
                    traceback.print_exc(file=f)
                last_log_time = time.time()

        time.sleep(screen_refresh_seconds)