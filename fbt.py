import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder

from scipy.spatial.transform import Rotation as R

def get_video_input(input_value):
    if input_value.isnumeric():
        print("using camera %s as input device..." % input_value)
        return int(input_value)

    print("using video '%s' as input..." % input_value)
    return input_value

def draw_pose_rect(image, rect, color=(255, 0, 255), thickness=2):
    image_width = image.shape[1]
    image_height = image.shape[0]

    world_rect = [(rect.x_center * image_width, rect.y_center * image_height),
                  (rect.width * image_width, rect.height * image_height),
                  rect.rotation]

    box = cv2.boxPoints(world_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, thickness)

def mediapipeTo3dpose(lms):
    #33 pose landmarks as in https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
    #convert landmarks returned by mediapipe to skeleton that I use.
    #lms = results.pose_world_landmarks.landmark

    pose = np.zeros((29,3))

    pose[0]=[lms[28].x,lms[28].y,lms[28].z]
    pose[1]=[lms[26].x,lms[26].y,lms[26].z]
    pose[2]=[lms[24].x,lms[24].y,lms[24].z]
    pose[3]=[lms[23].x,lms[23].y,lms[23].z]
    pose[4]=[lms[25].x,lms[25].y,lms[25].z]
    pose[5]=[lms[27].x,lms[27].y,lms[27].z]

    pose[6]=[0,0,0]

    #some keypoints in mediapipe are missing, so we calculate them as avarage of two keypoints
    pose[7]=[lms[12].x/2+lms[11].x/2,lms[12].y/2+lms[11].y/2,lms[12].z/2+lms[11].z/2]
    pose[8]=[lms[10].x/2+lms[9].x/2,lms[10].y/2+lms[9].y/2,lms[10].z/2+lms[9].z/2]

    pose[9]=[lms[0].x,lms[0].y,lms[0].z]

    pose[10]=[lms[15].x,lms[15].y,lms[15].z]
    pose[11]=[lms[13].x,lms[13].y,lms[13].z]
    pose[12]=[lms[11].x,lms[11].y,lms[11].z]

    pose[13]=[lms[12].x,lms[12].y,lms[12].z]
    pose[14]=[lms[14].x,lms[14].y,lms[14].z]
    pose[15]=[lms[16].x,lms[16].y,lms[16].z]

    pose[16]=[pose[6][0]/2+pose[7][0]/2,pose[6][1]/2+pose[7][1]/2,pose[6][2]/2+pose[7][2]/2]

    #right foot
    pose[17] = [lms[31].x,lms[31].y,lms[31].z]  #forward
    pose[18] = [lms[29].x,lms[29].y,lms[29].z]  #back
    pose[19] = [lms[25].x,lms[25].y,lms[25].z]  #up

    #left foot
    pose[20] = [lms[32].x,lms[32].y,lms[32].z]  #forward
    pose[21] = [lms[30].x,lms[30].y,lms[30].z]  #back
    pose[22] = [lms[26].x,lms[26].y,lms[26].z]  #up

    #right hand
    pose[23] = [lms[17].x,lms[17].y,lms[17].z]  #forward
    pose[24] = [lms[15].x,lms[15].y,lms[15].z]  #back
    pose[25] = [lms[19].x,lms[19].y,lms[19].z]  #up

    #left hand
    pose[26] = [lms[18].x,lms[18].y,lms[18].z]  #forward
    pose[27] = [lms[16].x,lms[16].y,lms[16].z]  #back
    pose[28] = [lms[20].x,lms[20].y,lms[20].z]  #up

    return pose

def get_rot(pose3d):
    ## guesses
    hip_left = 2
    hip_right = 3
    hip_up = 16

    knee_left = 1
    knee_right = 4

    ankle_left = 0
    ankle_right = 5

    # hip

    x = pose3d[hip_right] - pose3d[hip_left]
    w = pose3d[hip_up] - pose3d[hip_left]
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    hip_rot = np.vstack((x, y, z)).T

    # right leg

    y = pose3d[knee_right] - pose3d[ankle_right]
    w = pose3d[hip_right] - pose3d[ankle_right]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_left] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    leg_r_rot = np.vstack((x, y, z)).T

    # left leg

    y = pose3d[knee_left] - pose3d[ankle_left]
    w = pose3d[hip_left] - pose3d[ankle_left]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_right] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    leg_l_rot = np.vstack((x, y, z)).T

    rot_hip = R.from_matrix(hip_rot).as_quat()
    rot_leg_r = R.from_matrix(leg_r_rot).as_quat()
    rot_leg_l = R.from_matrix(leg_l_rot).as_quat()

    return rot_hip, rot_leg_l, rot_leg_r

def get_rot_mediapipe(pose3d):
    hip_left = pose3d[2]
    hip_right = pose3d[3]
    hip_up = pose3d[16]

    foot_l_f = pose3d[20]
    foot_l_b = pose3d[21]
    foot_l_u = pose3d[22]

    foot_r_f = pose3d[17]
    foot_r_b = pose3d[18]
    foot_r_u = pose3d[19]

    # hip

    x = hip_right - hip_left
    w = hip_up - hip_left
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    hip_rot = np.vstack((x, y, z)).T

    # left foot

    x = foot_l_f - foot_l_b
    w = foot_l_u - foot_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    l_foot_rot = np.vstack((x, y, z)).T

    # right foot

    x = foot_r_f - foot_r_b
    w = foot_r_u - foot_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    r_foot_rot = np.vstack((x, y, z)).T

    hip_rot = R.from_matrix(hip_rot).as_quat()
    r_foot_rot = R.from_matrix(r_foot_rot).as_quat()
    l_foot_rot = R.from_matrix(l_foot_rot).as_quat()

    return hip_rot, r_foot_rot, l_foot_rot

def get_rot_hands(pose3d):

    hand_r_f = pose3d[26]
    hand_r_b = pose3d[27]
    hand_r_u = pose3d[28]

    hand_l_f = pose3d[23]
    hand_l_b = pose3d[24]
    hand_l_u = pose3d[25]

    # left hand

    x = hand_l_f - hand_l_b
    w = hand_l_u - hand_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    l_hand_rot = np.vstack((z, y, -x)).T

    # right hand

    x = hand_r_f - hand_r_b
    w = hand_r_u - hand_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)

    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))

    r_hand_rot = np.vstack((z, y, -x)).T

    r_hand_rot = R.from_matrix(r_hand_rot).as_quat()
    l_hand_rot = R.from_matrix(l_hand_rot).as_quat()

    return l_hand_rot, r_hand_rot

def osc_build_msg(name, position_or_rotation, args):
    builder = osc_message_builder.OscMessageBuilder(address=f"/tracking/trackers/{name}/{position_or_rotation}")
    builder.add_arg(float(args[0]))
    builder.add_arg(float(args[1]))
    builder.add_arg(float(args[2]))
    return builder.build()

def osc_build_bundle(trackers):
    builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    builder.add_content(osc_build_msg(trackers[0]['name'], "position", trackers[0]['position']))
    for tracker in trackers[1:]:
        builder.add_content(osc_build_msg(tracker['name'], "position", tracker['position']))
        builder.add_content(osc_build_msg(tracker['name'], "rotation", tracker['rotation']))
    return builder.build()

def updatepose(client, pose3d, rots, hand_rots):

    headsetpos = [float(0),float(0),float(0)]
    headsetrot = R.from_quat([float(0),float(0),float(0),float(1)])

    neckoffset = headsetrot.apply([0,-0.2,0.1])

    pose3d = pose3d

    offset = pose3d[7] - (headsetpos+neckoffset)

    trackers = []
    trackers.append({ "name": "head", "position": [ 0, 0, 0 ]})

    for i in [(0,1),(5,2),(6,0)]:

        position = pose3d[i[0]] - offset

        rotation = R.from_quat(rots[i[1]])

        rotation = rotation.as_euler("ZXY", degrees=True)
        rotation = [ rotation[0], rotation[2], -rotation[1] ]
        trackers.append({ "name": str(i[1]+1), "position": position, "rotation": rotation })

    if len(trackers) > 1:
        client.send(osc_build_bundle(trackers))

    return True

client = udp_client.SimpleUDPClient("127.0.0.1", 9000, True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
        smooth_landmarks=True,
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5,
        #static_image_mode=,
        model_complexity=0)
cap = cv2.VideoCapture(get_video_input("0"))

connections = mp_pose.POSE_CONNECTIONS

rotation = 0
i = 0

use_hands = True
feet_rotation = True

euler_rot_y = 180
euler_rot_x = 90
euler_rot_z = 180

global_rot_y = R.from_euler('y',euler_rot_y,degrees=True)     #default rotations, for 0 degrees around y and x
global_rot_x = R.from_euler('x',euler_rot_x-90,degrees=True)
global_rot_z = R.from_euler('z',euler_rot_z-180,degrees=True)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = pose.process(image)

    if results.pose_world_landmarks:        #if any pose was detected
            pose3d = mediapipeTo3dpose(results.pose_world_landmarks.landmark)   #convert keypoints to a format we use

            #do we need this with osc as well?

            pose3d[:,0] = -pose3d[:,0]      #flip the points a bit since steamvrs coordinate system is a bit diffrent
            pose3d[:,1] = -pose3d[:,1]

            pose3d_og = pose3d.copy()
            #params.pose3d_og = pose3d_og

            for j in range(pose3d.shape[0]):        #apply the rotations from the sliders
                pose3d[j] = global_rot_z.apply(pose3d[j])
                pose3d[j] = global_rot_x.apply(pose3d[j])
                pose3d[j] = global_rot_y.apply(pose3d[j])

            if not feet_rotation:
                rots = get_rot(pose3d)          #get rotation data of feet and hips from the position-only skeleton data
            else:
                rots = get_rot_mediapipe(pose3d)

            if use_hands:
                hand_rots = get_rot_hands(pose3d)
            else:
                hand_rots = None

            if not updatepose(client, pose3d, rots, hand_rots):
                continue

    if hasattr(results, "pose_rect_from_landmarks"):
        draw_pose_rect(image, results.pose_rect_from_landmarks)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, connections)
    cv2.imshow('MediaPipe OSC Pose', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
