import threading
from PIL import Image
from utils import *

ROUTE_ID = "Green-D"
INBOUND_DIRECTION_ID = 1
START_STOP_ID = "place-river"
TARGET_STOP_ID = "place-bcnfd"
SCREEN_REFRESH_SECONDS = 0.01
SCROLL_SPEED = 3
IMG_MAP = Image.open(f"{ROOT_DIR}map_gl.png").convert("RGBA")
TOP_LEFT_POS = (42.367667, -71.262748)
BOTTOM_RIGHT_POS = (42.314685, -71.122894)
LANDMARKS = [((42.336868, -71.253381), "Riverside"), ((42.332698, -71.243147), "Woodland"), ((42.325722, -71.230420), "Waban"), ((42.319153, -71.216586), "Eliot"), ((42.322388, -71.205483), "Newton Highlands"), ((42.329449, -71.192388), "Newton Centre"), ((42.326727, -71.164689), "Chestnut Hill"), ((42.335111, -71.148712),"Reservoir"), ((42.335788, -71.140449),"Beaconsfield"), ((42.336630, -71.141131), "House")]
STOP_SEQUENCES = [390, 380, 370, 360, 350, 340, 330, 320, 310]
# Eliot is 50
# Newton Highlands is 30
# Newton Centre is 20
# Chestnut hill is 10
# Reservoir is 10
ZOOM = 8

DEBUG = False
DEBUG_COORDINATES = [(42.335788, -71.140449)]

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

    main(STOP_SEQUENCES, SCREEN_REFRESH_SECONDS, SCROLL_SPEED, "TRAIN", DEBUG, DEBUG_COORDINATES)

