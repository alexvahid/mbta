import threading
from PIL import Image
from utils import *

ROUTE_ID = "93"
INBOUND_DIRECTION_ID = 1
START_STOP_ID = "29010"
TARGET_STOP_ID = "2845"
TARGET_STOP_SEQUENCE = 4
SCREEN_REFRESH_SECONDS = 0.01
SCROLL_SPEED = 2
IMG_MAP = Image.open(f"{ROOT_DIR}map.png").convert("RGBA")
TOP_LEFT_POS = (42.390035, -71.084619)
BOTTOM_RIGHT_POS = (42.372526, -71.060660)
LANDMARKS = [((42.384390, -71.075927), "Sullivan Sq."), ((42.382510, -71.069484), "Home")]
STOP_SEQUENCES = [4,3,2,1,0]
ZOOM = 3

DEBUG = False
DEBUG_COORDINATES = [(42.383912, -71.073844),(42.384097, -71.072975),(42.383764, -71.072407)]

if __name__ == "__main__":
    prediction_thread = threading.Thread(
        target=prediction_data_refresher, 
        args=(ROUTE_ID, TARGET_STOP_ID, INBOUND_DIRECTION_ID),
        daemon=True)
    prediction_thread.start()

    vehicle_thread = threading.Thread(
        target=vehicle_data_refresher,
        args=(ROUTE_ID, INBOUND_DIRECTION_ID, START_STOP_ID, STOP_SEQUENCES, EMPTY_IMAGE, DEBUG, DEBUG_COORDINATES),
        daemon=True
    )
    vehicle_thread.start()

    background_image_thread = threading.Thread(
        target=background_refresher, 
        args=(IMG_MAP, TOP_LEFT_POS, BOTTOM_RIGHT_POS, LANDMARKS, ZOOM, DEBUG, DEBUG_COORDINATES),
        daemon=True
    )
    background_image_thread.start()

    main(STOP_SEQUENCES, SCREEN_REFRESH_SECONDS, SCROLL_SPEED, "BUS", DEBUG, DEBUG_COORDINATES)
