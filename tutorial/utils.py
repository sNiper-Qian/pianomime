#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import collections
from robopianist.models.hands import shadow_hand_constants as hand_consts
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist.suite.tasks import piano_with_one_shadow_hand
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from robopianist.models.hands import HandSide
from robopianist.models.piano import piano_constants as consts
import matplotlib.pyplot as plt
import math
from dm_control.mujoco.wrapper import mjbindings
from collections import defaultdict

mjlib = mjbindings.mjlib

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
COORDINATE_TEXT_COLOR = (88, 205, 54) # vibrant green
HANDEDNESS_TEXT_COLOR = (255, 0, 0) # blue  
KEY_TEXT_COLOR = (0, 0, 255) # red
WIDTH = 3420
HEIGHT = 2214
NUM_WHITE_KEYS = 52
PIANO_LOWER_BOUND = 1540
PIANO_UPPER_BOUND = 1100
WHITE_KEY_INDICES = [
        0,
        2,
        3,
        5,
        7,
        8,
        10,
        12,
        14,
        15,
        17,
        19,
        20,
        22,
        24,
        26,
        27,
        29,
        31,
        32,
        34,
        36,
        38,
        39,
        41,
        43,
        44,
        46,
        48,
        50,
        51,
        53,
        55,
        56,
        58,
        60,
        62,
        63,
        65,
        67,
        68,
        70,
        72,
        74,
        75,
        77,
        79,
        80,
        82,
        84,
        86,
        87,
    ]

BLACK_TWIN_KEY_INDICES = [
        4,
        6,
        16,
        18,
        28,
        30,
        40,
        42,
        52,
        54,
        64,
        66,
        76,
        78,
    ]
BLACK_TRIPLET_KEY_INDICES = [
        1,
        9,
        11,
        13,
        21,
        23,
        25,
        33,
        35,
        37,
        45,
        47,
        49,
        57,
        59,
        61,
        69,
        71,
        73,
        81,
        83,
        85,
    ]

# Named tuple:(x, y, key_pressed(-1 means no key pressed)):
#   x: x coordinate of the hand
#   y: y coordinate of the hand
#   key_pressed: the key pressed by the hand, -1 means no key pressed
# Order: Wrist, Thumb, Index, Middle, Ring, Pinky
# Break quats into four parts for easier saving with numpy
HandAction = collections.namedtuple('HandAction', ['xs', 'ys', 'zs', 'quat0', 'quat1', 'quat2', 'quat3', 'keys_pressed'])

task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
        hand_side=HandSide.LEFT,
        midi=music.load("TwinkleTwinkleRousseau"),
        disable_colorization=True,
        change_color_on_activation=True,
        trim_silence=True,
        control_timestep=0.01,
        )

env = composer_utils.Environment(
    recompile_physics=False, task=task, strip_singleton_obs_buffer_dim=True
)

def pixel2mujoco_coordinate(pixel_point, H):
  pixel_point = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
  world_point = cv2.perspectiveTransform(pixel_point, H)
  mujoco_point = (-world_point[0][0][1], world_point[0][0][0])
  return mujoco_point

def mujo2pixel_coordinate(mujoco_point, H):
  world_point = np.array([[[mujoco_point[1], -mujoco_point[0]]]], dtype=np.float32)
  # Inverse the homography matrix
  H_inv = np.linalg.inv(H)  
  pixel_point = cv2.perspectiveTransform(world_point, H_inv)
  return pixel_point[0][0]

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Adjust the brightness and contrast of an image.
    Alpha (contrast): >1 increases contrast, <1 decreases contrast.
    Beta (brightness): >0 increases brightness, <0 decreases brightness.
    """
    # New image = alpha * original image + beta
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def create_detector():
  model_path = "./hand_landmark.task"
  BaseOptions = mp.tasks.BaseOptions
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options,
                                        running_mode = mp.tasks.vision.RunningMode.VIDEO,
                                        num_hands=2)
  detector = HandLandmarker.create_from_options(options)
  return detector

def get_duplicate_indices(lst):
    indices = defaultdict(lst)
    duplicates = {}

    for index, item in enumerate(lst):
        indices[item].append(index)
        if len(indices[item]) > 1:
            duplicates[item] = indices[item]

    return duplicates

def draw_horizontal_line(img, y):
    # Copy the original image to avoid modifying it
    img_with_line = img.copy()
    
    # Get the width of the image
    width = img.shape[1]
    
    # Set the color of the line (B, G, R) and thickness
    color = (0, 255, 0)  # Green color
    thickness = 2  # Line thickness
    
    # Draw a horizontal line at y-coordinate 'y'
    start_point = (0, y)
    end_point = (width, y)
    img_with_line = cv2.line(img_with_line, start_point, end_point, color, thickness)
    
    return img_with_line

def draw_vertical_lines(img, interval=WIDTH / NUM_WHITE_KEYS):
    # Copy the original image to avoid modifying it
    img_with_lines = img.copy()
    
    # Get the dimensions of the image
    height, width = img.shape[:2]
    
    # Set the color of the line (B, G, R) and thickness
    color = (0, 255, 0)  # Green color
    thickness = 2  # Line thickness
    
    # Draw vertical lines every 'interval' pixels
    key_widths = [54, 62, 63, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 67, 69, 71, 72, 69, 69, 68, 66, 66, 64, 63, 63, 63]
    x = 0
    for w in key_widths:
        x += w
        x_int = int(x)
        start_point = (x_int, 0)
        end_point = (x_int, height)
        img_with_lines = cv2.line(img_with_lines, start_point, end_point, color, thickness)
    
    return img_with_lines

def draw_keys_on_image(rgb_image, H, keys, timestep=None):
    annotated_image = rgb_image.copy()
    for key in keys:
        key_pos = env.task.piano._keys[key].pos
        pixel_pos = mujo2pixel_coordinate(key_pos, H)
        # Draw the key number
        cv2.putText(annotated_image, f"{key}",
                  (int(pixel_pos[0]), int(pixel_pos[1])), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, KEY_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    if timestep:
      #  Put time step on the left top corner
      cv2.putText(annotated_image, f"{timestep}",
                (0, 50), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, COORDINATE_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image

def draw_point_on_image(rgb_image, H, points):
    annotated_image = rgb_image.copy()
    for point in points:
      pixel_pos = mujo2pixel_coordinate(point, H)
      # draw the point on the image
      cv2.circle(annotated_image, (int(pixel_pos[0]), int(pixel_pos[1])), 5, (0, 0, 255), -1)
    return annotated_image

def draw_landmarks_on_image(rgb_image, detection_result, timestep=None):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def extract_finger_based_on_key(rgb_image, detection_result, keys, H, last_fingering=None, last_keys=None):
  hand_world_landmarks_list = detection_result.hand_world_landmarks
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = [handedness[0].category_name for handedness in detection_result.handedness]
  annotated_image = rgb_image.copy()
  height, width, _ = rgb_image.shape
  # Determine the handedness
  # Default is left hand first, right hand second
  if len(handedness_list) == 0:
    pass
  elif len(handedness_list) == 1:
    # If little finger is on the left side of thumb, then it is left hand
    if hand_landmarks_list[0][20].x < hand_landmarks_list[0][4].x:
        handedness_list[0] = 'Left'
    else:
        handedness_list[0] = 'Right'
  else:
    handedness_list[0] = 'Left'
    handedness_list[1] = 'Right'
    # Compare the x coordinate of the wrist
    if hand_landmarks_list[0][0].x > hand_landmarks_list[1][0].x:
      # Reverse the order of the hand landmarks
      hand_landmarks_list[0], hand_landmarks_list[1] = hand_landmarks_list[1], hand_landmarks_list[0]
  # Map from finger index in handtracking model to index in midi
  left_finger_idx_map = {0: -1, 4: 5, 8: 6, 12: 7, 16: 8, 20: 9}
  right_finger_idx_map = {0: -1, 4: 0, 8: 1, 12: 2, 16: 3, 20: 4}
  # Map from index in midi to finger index in handtracking model
  left_finger_idx_map_inv = {v: k for k, v in left_finger_idx_map.items()}
  right_finger_idx_map_inv = {v: k for k, v in right_finger_idx_map.items()}
  
  finger_pos_dict = {}
  if last_keys and last_fingering and keys == last_keys:
    keys = last_keys
    fingering = last_fingering
  else:
    for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
        for finger_idx, landmark in enumerate(hand_landmarks):
          if finger_idx % 4 != 0 or finger_idx == 0:
            continue
          x = int(landmark.x * width)
          y = int(landmark.y * height)
          mujoco_point = pixel2mujoco_coordinate((x, y), H)
          if handedness == 'Left':
            finger_pos_dict[left_finger_idx_map[finger_idx]] = mujoco_point
          else:
            finger_pos_dict[right_finger_idx_map[finger_idx]] = mujoco_point

    fingering = []
    candidates = {}
    occupied_fingers = []
    candidtated_times = {}
    for key in keys:
      key_pos = env.task.piano._keys[key].pos
      if last_fingering and last_keys and key in last_keys:
         fingering.append(last_fingering[last_keys.index(key)])
         occupied_fingers.append(last_fingering[last_keys.index(key)])
      else:
        min_dist = 100
        min_dist_finger_idx = -1
        # Find the closest finger to the key
        try:
          if key <= 20:
            if key in WHITE_KEY_INDICES and 9 not in occupied_fingers \
                and abs(finger_pos_dict[9][1] - key_pos[1]) < 1.5*consts.WHITE_KEY_WIDTH:
              fingering.append(9)
              occupied_fingers.append(9)
              continue
            elif key in (BLACK_TWIN_KEY_INDICES or key in BLACK_TRIPLET_KEY_INDICES) and 8 not in occupied_fingers \
                and abs(finger_pos_dict[8][1] - key_pos[1]) < 1.5*consts.WHITE_KEY_WIDTH:
              fingering.append(8)
              occupied_fingers.append(8)
              continue
        except:
           pass
        for finger_idx, finger_pos in finger_pos_dict.items():
          if finger_idx in occupied_fingers:
            continue
          dist = abs(finger_pos[1] - key_pos[1])
          if dist > 2*consts.WHITE_KEY_WIDTH:
            continue
          if dist < min_dist and finger_pos[0] < key_pos[0] + 0.52 * consts.WHITE_KEY_LENGTH: # Sometimes the thumb is out of the keyboard
            min_dist = dist
            min_dist_finger_idx = finger_idx
        if min_dist_finger_idx == -1:
            # Ask the user to choose the finger
            print("Last keys:", last_keys)
            print("All keys:", keys)
            print("Please choose the finger for key {}".format(key))
            print("Candidate fingers:", candidates)
            user_finger_idx = int(input())
            while user_finger_idx < 0 or user_finger_idx > 9:
              print("Invalid finger index! Please choose again.")
              user_finger_idx = int(input())
            fingering.append(user_finger_idx)
            occupied_fingers.append(user_finger_idx)
        else:
          fingering.append(min_dist_finger_idx)
          occupied_fingers.append(min_dist_finger_idx)
    # Split the fingering into left and right hand (0-4 is right hand, 5-9 is left hand)
    left_fingering = []
    right_fingering = []
    for finger in fingering:
      if finger > 4:
        left_fingering.append(finger)
      else:
        right_fingering.append(finger)
    # Sort the left_fingering from big to small
    left_fingering[:3].sort(reverse=True)
    # Sort the right_fingering from small to big
    right_fingering[2:].sort()
    fingering = left_fingering + right_fingering
  try:
    for i, finger in enumerate(fingering):
      if finger > 4:
        finger_idx = left_finger_idx_map_inv[finger]
        x = int(hand_landmarks_list[0][finger_idx].x * width)
        y = int(hand_landmarks_list[0][finger_idx].y * height)
        cv2.putText(annotated_image, f"{keys[i]}",
                  (x, y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, KEY_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
      else:
        finger_idx = right_finger_idx_map_inv[finger]
        x = int(hand_landmarks_list[1][finger_idx].x * width)
        y = int(hand_landmarks_list[1][finger_idx].y * height)
        cv2.putText(annotated_image, f"{keys[i]}",
                  (x, y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, KEY_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
  except:
    pass
  return annotated_image, fingering
        
def process_landmarks(rgb_image, detection_result, keys=None, fingering=None, H=None, timestep=None):
  hand_world_landmarks_list = detection_result.hand_world_landmarks
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = [handedness[0].category_name for handedness in detection_result.handedness]

  # Determine the handedness
  # Default is left hand first, right hand second
  if len(handedness_list) == 0:
    return rgb_image, [], []
  elif len(handedness_list) == 1:
    # If little finger is on the left side of thumb, then it is left hand
    if hand_landmarks_list[0][20].x < hand_landmarks_list[0][4].x:
        handedness_list[0] = 'Left'
    else:
        handedness_list[0] = 'Right'
  else:
    handedness_list[0] = 'Left'
    handedness_list[1] = 'Right'
    # Compare the x coordinate of the wrist
    if hand_landmarks_list[0][0].x > hand_landmarks_list[1][0].x:
      # Reverse the order of the hand landmarks
      hand_landmarks_list[0], hand_landmarks_list[1] = hand_landmarks_list[1], hand_landmarks_list[0]
    
  annotated_image = np.copy(rgb_image)
  hand_action_list = []
  # Map from finger index in handtracking model to index in midi
  left_finger_idx_map = {0: -1, 4: 5, 8: 6, 12: 7, 16: 8, 20: 9}
  right_finger_idx_map = {0: -1, 4: 0, 8: 1, 12: 2, 16: 3, 20: 4}
  # Map from index in midi to finger index in handtracking model
  left_finger_idx_map_inv = {v: k for k, v in left_finger_idx_map.items()}
  right_finger_idx_map_inv = {v: k for k, v in right_finger_idx_map.items()}
  
  # Loop through the detected hands to visualize.
  for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]

    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # if keys is not None:
      # # Make a string consists of key (a list)
      # key_str = ""
      # for key in keys:
      #   key_str += str(key)
      #   key_str += " "

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    if keys is not None and fingering is not None and H is not None:
      # Get HandAction
      # Get the joint cartesian coordinates
      xs = []
      ys = []
      for finger_idx, landmark in enumerate(hand_landmarks):
        if finger_idx % 4 != 0:
          continue
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        mujoco_point = pixel2mujoco_coordinate((x, y), H)
        xs.append(mujoco_point[0])
        ys.append(mujoco_point[1])
        cv2.putText(annotated_image, "{:.3f}, {:.3f}".format(mujoco_point[0], mujoco_point[1]),
                  (x, y+10), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, COORDINATE_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)  

      xs = np.array(xs)
      ys = np.array(ys)

      # Get the key pressed by the hand
      # And add the z coordinate to HandAction
      keys_pressed = np.ones(6) * -1 # -1 means no key pressed
      zs = np.ones(6) * consts.WHITE_KEY_HEIGHT * 2
      for i, finger in enumerate(fingering):
          if finger > 4 and handedness == 'Left': # left hand
            key = keys[i]
            finger_idx = left_finger_idx_map_inv[finger]
            keys_pressed[finger_idx//4] = key
            ys[finger_idx//4] = env.task.piano._keys[key].pos[1]
            if key in WHITE_KEY_INDICES:
              zs[finger_idx//4] = 0
            elif key in BLACK_TWIN_KEY_INDICES or key in BLACK_TRIPLET_KEY_INDICES:
              zs[finger_idx//4] = 0
          elif finger <= 4 and handedness == 'Right': # right hand
            key = keys[i]
            finger_idx = right_finger_idx_map_inv[finger]
            keys_pressed[finger_idx//4] = key
            ys[finger_idx//4] = env.task.piano._keys[key].pos[1]
            if key in WHITE_KEY_INDICES:
              zs[finger_idx//4] = 0
            elif key in BLACK_TWIN_KEY_INDICES or key in BLACK_TRIPLET_KEY_INDICES:
              zs[finger_idx//4] = 0
          else:
            continue
          pixel_x = hand_landmarks[finger_idx].x * width
          pixel_y = hand_landmarks[finger_idx].y * height
          cv2.putText(annotated_image, str(key), (int(pixel_x), int(pixel_y)-10), 
                      cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, KEY_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
      
      # Get the quaternion of the hand
      quats = []
      for i in range(6):
          if i < 6: # Dummy quaternion
            quats.append([1, 0, 0, 0])
          else:
            # Do not use anymore
            parent_x = hand_landmarks[i*4-1].x * width
            parent_y = hand_landmarks[i*4-1].y * height
            pparent_x = hand_landmarks[i*4-2].x * width
            pparent_y = hand_landmarks[i*4-2].y * height
            parent_mujoco = pixel2mujoco_coordinate((parent_x, parent_y), H)
            pparent_mujoco = pixel2mujoco_coordinate((pparent_x, pparent_y), H)
            # Z axis of finger tip is towards the child
            delta_x = xs[i] - parent_mujoco[0]
            delta_y = ys[i] - parent_mujoco[1]
            delta_z = -math.sqrt(hand_consts.TIP_LENGTHS[i-1] ** 2 - delta_x ** 2 - delta_y ** 2)
            z_axis = np.array([delta_x, delta_y, delta_z])
            # X axis of finger tip is vertical to the plane of child and parent and parent and its parent
            delta_x_parent = parent_mujoco[0] - pparent_mujoco[0]
            delta_y_parent = parent_mujoco[1] - pparent_mujoco[1]
            delta_z_parent = -math.sqrt(hand_consts.PARENT_LENGTHS[i-1] ** 2 - delta_x_parent ** 2 - delta_y_parent ** 2)
            if i == 1: # Thumb
              delta_z_parent = -delta_z_parent # Thumb tip's parent body is lower than the tip
            parent_z_axis = np.array([delta_x_parent, delta_y_parent, delta_z_parent])
            x_axis = np.cross(z_axis, parent_z_axis)
            # For left hand, x axis should have negative y component
            # TODO: check if this is correct for right hand
            if x_axis[1] > 0:
              x_axis = -x_axis
            # Normalize the axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            # Rotation matrix
            rot_mat = np.column_stack((x_axis, y_axis, z_axis))
            quat = np.zeros(4)
            mjlib.mju_mat2Quat(quat, rot_mat.flatten())
              # print("z_axis:", z_axis)
              # print("parent_z_axis:", parent_z_axis)
            # print(i)
            # print("rot_mat:", rot_mat)
            # print("quat:", quat)
            quats.append(quat)
            pixel_x = hand_landmarks[i*4].x * width
            pixel_y = hand_landmarks[i*4].y * height
            # cv2.putText(annotated_image, "{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(quat[0], quat[1], quat[2], quat[3]), (int(pixel_x), int(pixel_y)-10), 
              # cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, KEY_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
      quats = np.array(quats)
      # Break quats into 4 lists
      quats = np.split(quats, 4, axis=1)
      quats = [np.squeeze(quat) for quat in quats]

      hand_action = HandAction(xs, ys, zs, quats[0], quats[1], quats[2], quats[3], keys_pressed)
      hand_action_list.append(hand_action)
  # if timestep:
  #   #  Put time step on the left top corner
  #   cv2.putText(annotated_image, f"{timestep}",
  #             (50, 50), cv2.FONT_HERSHEY_DUPLEX,
  #             FONT_SIZE, COORDINATE_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # annotated_image = draw_horizontal_line(annotated_image, PIANO_LOWER_BOUND)
    # annotated_image = draw_horizontal_line(annotated_image, PIANO_UPPER_BOUND)

    # # Draw the world landmarks
    # # only enumerate the 4th, 8th, 12th, 16th, 20th landmarks
    # for idx, landmark in enumerate(hand_world_landmarks):
    #   if idx % 4 != 0:
    #     continue
    #   landmark_x = format(landmark.x*100, '.1f')
    #   landmark_y = format(landmark.y*100, '.1f')
    #   landmark_z = format(landmark.z*100, '.1f')
    #   cv2.putText(annotated_image, f"{landmark_x}, {landmark_y}, {landmark_z}",
    #                (int(x_coordinates[idx]*width), int(y_coordinates[idx]*height)), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
  return annotated_image, hand_action_list, handedness_list

def enhance_hand_visibility(img):
    # Convert image to YCrCb color space
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    
    # Enhance the contrast of the Y channel (brightness)
    y = cv2.equalizeHist(y)
    
    # Merge back the channels
    enhanced_img = cv2.merge([y, cr, cb])
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_YCrCb2BGR)
    
    # Optional: Apply sharpening filter to enhance edges
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
    
    return enhanced_img


def preprocess_frame(frame):
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Apply histogram equalization to the Y channel
    y_eq = cv2.equalizeHist(y)
    
    # Apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    
    # Merge back the channels
    frame_eq = cv2.merge([y_eq, cr, cb])
    frame_eq = cv2.cvtColor(frame_eq, cv2.COLOR_YCrCb2BGR)
    
    frame_clahe = cv2.merge([y_clahe, cr, cb])
    frame_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_YCrCb2BGR)
    
    return frame_eq, frame_clahe

def adjust_hand_action(hand_action, last_hand_action=None):
    for i in range(6):
       if hand_action.keys_pressed[i] != -1:
          key = int(hand_action.keys_pressed[i])
          key_x = env.task.piano._keys[key].pos[0]
          key_y = env.task.piano._keys[key].pos[1]
          x = hand_action.xs[i]
          y = hand_action.ys[i]
          if key in WHITE_KEY_INDICES:
            if not(x < key_x + consts.WHITE_KEY_LENGTH*0.48 and x > key_x - consts.WHITE_KEY_LENGTH*0.48) or \
              not(y < key_y + consts.WHITE_KEY_WIDTH*0.48 and y > key_y - consts.WHITE_KEY_WIDTH*0.48):
              if last_hand_action and last_hand_action.keys_pressed[i] != -1:
                # If the last action is valid, use the last action
                hand_action.xs[i] = last_hand_action.xs[i]
                hand_action.ys[i] = last_hand_action.ys[i]
              else:
                # If the last action is invalid, use the key position
                hand_action.xs[i] = key_x + 1/4*consts.WHITE_KEY_LENGTH # Press at 1/4 of the key length is a reasonable value
                hand_action.ys[i] = key_y
          elif key in BLACK_TWIN_KEY_INDICES or key in BLACK_TRIPLET_KEY_INDICES:
             if not(x < key_x + consts.BLACK_KEY_LENGTH*0.48 and x > key_x - consts.BLACK_KEY_LENGTH*0.48) or \
              not(y < key_y + consts.BLACK_KEY_WIDTH*0.48 and y > key_y - consts.BLACK_KEY_WIDTH*0.48):
              if last_hand_action and last_hand_action.keys_pressed[i] != -1:
                # If the last action is valid, use the last action
                hand_action.xs[i] = last_hand_action.xs[i]
                hand_action.ys[i] = last_hand_action.ys[i]
              else:
                # If the last action is invalid, use the key position
                hand_action.xs[i] = key_x + 1/4*consts.BLACK_KEY_LENGTH
                hand_action.ys[i] = key_y
    return hand_action

            
def val_hand_action(hand_action):
    for i in range(6):
       if hand_action.keys_pressed[i] != -1:
          key = int(hand_action.keys_pressed[i])
          key_x = env.task.piano._keys[key].pos[0]
          key_y = env.task.piano._keys[key].pos[1]
          x = hand_action.xs[i]
          y = hand_action.ys[i]
          if key in WHITE_KEY_INDICES:
            length = consts.WHITE_KEY_LENGTH
            width = consts.WHITE_KEY_WIDTH
          elif key in BLACK_TWIN_KEY_INDICES or key in BLACK_TRIPLET_KEY_INDICES:
            length = consts.BLACK_KEY_LENGTH
            width = consts.BLACK_KEY_WIDTH
          if not(x < key_x + length/2 and x > key_x - length/2):
            print("key:", key)
            print("x:", x)
            print("key_x", key_x)
          if not(y < key_y + width/2 and y > key_y - width/2):
            print("key:", key)
            print("y:", y)
            print("key_y", key_y)

def add_z_to_hand_action(hand_action_list):
    for action in hand_action_list:
        action.xs = np.array(action.xs)
        action.ys = np.array(action.ys)
        action.zs = np.zeros(6)

if __name__ == '__main__':
   H = np.load('handtracking/H_matrices/PianoX.npy')
   print(pixel2mujoco_coordinate((248, 1398), H))