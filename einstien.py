import os
import cv2
# import matplotlib
# matplotlib.use('QtAgg')  # or 'Qt5Agg', 'GTK3Agg' depending on what's available

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import openpifpaf
from transformers import pipeline
from ultralytics import YOLOWorld
from transformers import pipeline
from YOLO3D.inference_yolo3D import detect3d
import math
from infer_utils import* #draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from lane_detection import*
from MaskRCNN.class_names import INSTANCE_CATEGORY_NAMES as class_names
from humanpifpaf import*
from deticbridge import*
from speedsign import*
from arrow import*
from traffic_light import identify_traffic_light as identify_traffic_light
#from pose_correction import*
#from lane_vis import*
from lane_info import*
from RAFT.demo import raft_flow
import json
global bump_condition
global bumpoff_count
global prev_flow
# from color_detector import*
####path to image####
def detect_traffic_light_color(image, box):
    x1, y1, x2, y2 = map(int,box)  # Bounding box coordinates
    
    
    # Extract the region of interest (ROI) around the traffic light
    roi = image[y1:y2, x1:x2]
    
    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and green in HSV
    red_lower1, red_upper1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    
    yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([40, 255, 255])
    green_lower, green_upper = np.array([40, 50, 50]), np.array([90, 255, 255])

    # Split the ROI into three sections (top, middle, bottom)
    # Split the ROI into three sections (top, middle, bottom)
    height = roi.shape[0]
    section_height = height // 3
    
    def split_and_prune_hsv(hsv):
        height, width, _ = hsv.shape
        section_height = height // 3  # Split into 3 sections

        # Compute 20% margin for width pruning
        margin_x = int(0.3 * width)

        # Prune width and extract sections
        top = hsv[:section_height, :]
        middle = hsv[section_height: 2 * section_height, :]
        bottom = hsv[2 * section_height:,:]

        return top, middle, bottom

# Example usage:
    top, middle, bottom = split_and_prune_hsv(hsv)
    
    
    def count_color(section, lower, upper):
        mask = cv2.inRange(section, lower, upper)
        return cv2.countNonZero(mask)*100 / (section.shape[0] * section.shape[1])  # Normalize by area

    # Check color in each section
    red_top = count_color(top, red_lower1, red_upper1) + count_color(top, red_lower2, red_upper2)
    yellow_middle = count_color(middle, yellow_lower, yellow_upper)
    green_bottom = count_color(bottom, green_lower, green_upper)

    # Thresholds for high color presence
    threshold = 0.10  # At least 30% of the section must be of that color

    # Decision based on dominant color
    if red_top > threshold:
        return "Red", #color , shape
    elif yellow_middle > threshold:
        return "Yellow"
    elif green_bottom > threshold:
        return "Green"
    else:
        return "Unknown"


import cv2
import numpy as np

def get_bump_loc(image_path):
    """
    Loads a binary image from the given path and finds the vertical coordinate (v)
    of the first white pixel encountered by sliding a horizontal line upward 
    from the bottom center.

    Args:
        image_path (str): Path to the binary image (0 and 255 values).

    Returns:
        int: Vertical coordinate (v) of the first white pixel found along the scan line.
             Returns -1 if no white pixel is found.
    """
    # Load image in grayscale
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if binary_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    # cv2.imshow("Orientation",binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    h, w = binary_image.shape
    center_x = w // 2
    x_start = max(center_x - 10, 0)
    x_end = min(center_x + 10, w - 1)

    for v in range(h - 1, (h//2+1), -1):  # from bottom to top
        line = binary_image[v, x_start:x_end+1]
        if np.any(line == 255):
            return v

    return -1




def check_brakelight(image, bbox):
    """
    Check if the backlight is illuminated (based on brightness) within a bounding box.

    Args:
        image (numpy.ndarray): The input BGR image.
        bbox (tuple): A tuple of (x1, y1, x2, y2) bounding box.

    Returns:
        bool: True if the light is likely on (bright), otherwise False.
    """
    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2]

    if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
        return False
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Convert to HSV and extract the Value (brightness) channel
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    value_channel = hsv_patch[:, :, 2]

    # Threshold for bright pixels (you can tune this)
    bright_pixels = np.sum(value_channel > 200)
    total_pixels = value_channel.size
    bright_ratio = bright_pixels / total_pixels

    # Show patch for visual debugging
    
    # cv2.imshow("Keypoints Visualization", image)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()
    return bright_ratio > 0.2  # You can tweak this threshold as needed

def estimate_truck_orientation(image, x1, y1, x2, y2):
    """
    Estimate truck facing direction (0° or 180°) using contour analysis.
    
    - Shows the image with drawing.
    - Returns only the rotation (0 or 180).
    """
    vis_image = image.copy()

    # Crop bounding box
    box = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contour found.")
        cv2.imshow("Orientation", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    # Largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(largest_contour)
    contour_center_x = cx + cw // 2
    box_width = x2 - x1

    # Decide orientation
    if contour_center_x > box_width // 2:
        rotation = 180
        direction = "→"
    else:
        rotation = 0
        direction = "←"

    # Draw bounding box
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Draw contour (shifted)
    shifted_contour = [c + [x1, y1] for c in largest_contour]
    cv2.drawContours(vis_image, [np.array(shifted_contour)], -1, (0, 0, 255), 2)

    # Put label
    label = f"Rot: {rotation}° {direction}"
    cv2.putText(vis_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # # Show result
    # cv2.imshow("Orientation", vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return rotation

def box_to_pif(orient,prediction):
    keypoints = prediction.data.reshape(-1, 3)
    
    ##########################################################
    keylist=[]
    for j,(x,y,confidence) in enumerate(keypoints):
        if x>0 and y>0 and confidence>0.20:
            keylist.append(j+1)
    print("keylist",keylist)
    #########################which view is available####################new
    # orient=-1*orient
    # finalpose=pose_correction(keylist,orient)
    # return finalpose
    
    if 70<=orient<120:
    ### getting the list
        backset=set()
        backset=(11,12,13,14,15,16,21,22)
        lpoint=7
        rpoint=17
        bcond=False
        lmcond=False
        rmcond=False
        for f in keylist:
            if f in backset:
                bcond=True
            if f ==lpoint:
                lmcond=True
            if f ==rpoint:
                rmcond=True
        if bcond and not lmcond and not rmcond:
            return orient
        if bcond and lmcond:
            return orient+10
        if bcond and rmcond:
            return orient-10
        
        if lmcond:
            orient=orient+10
        if rmcond:
            orient=orient-10
        return -1*orient
    
    elif -110<=orient<-80:
        ### getting the list
        frontset=set()
        frontset=(1,2,3,4,5,6)
        lpoint=7
        rpoint=17
        fcond=False
        lmcond=False
        rmcond=False
        for f in keylist:
            if f in frontset:
                fcond=True
            if f ==lpoint:
                lmcond=True
            if f ==rpoint:
                rmcond=True
        if fcond and not lmcond and not rmcond:
            return orient
        if fcond and lmcond:
            return orient-10
        if fcond and rmcond:
            return orient+10 
        if lmcond:
            orient=orient-10
        if rmcond:
            orient=orient+10
        return -1*orient
    elif -80<=orient<70:
        orient=-1*orient
        lpoint=7
        for f in keylist:
            if f==lpoint:
                return orient+np.pi
        return orient
    
    elif -180<orient<-130 or 130<orient<180:
        orient=-1*orient
        rpoint=17
        for f in keylist:
            if f==rpoint:
                return orient+np.pi
        return orient
    
    else:
        return -1*orient
   
   
        


def run_inference(image_path, results, orients, output_path="./SelfDrivingViz/Code/car_output.jpg", checkpoint="shufflenetv2k16-apollo-24"):
    # Load model and image
    predictor = openpifpaf.Predictor(checkpoint=checkpoint)
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    image_vis=cv2.imread(image_path)
    box_orient=orients
    # Get predictions
    predictions, _, _ = predictor.numpy_image(image_np)
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    
    # Draw keypoints using OpenPifPaf's built-in painter
    painter = openpifpaf.show.AnnotationPainter()
    painter.annotations(ax, predictions)
    
    # Process each detected object
    key_minmax_dict={}
    for i, prediction in enumerate(predictions):
        keypoints = prediction.data.reshape(-1, 3)  # Reshape to (N, 3) -> (x, y, confidence)
        minx, miny = math.inf, math.inf
        maxx, maxy = 0, 0
        
        for x, y, confidence in keypoints:
            if x>0 and y>0:
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y

        key_minmax_dict[i]=(minx, miny, maxx, maxy)
    # Draw bounding boxes & update orientation
    orient_set=set()
    for i,(minx, miny, maxx, maxy) in key_minmax_dict.items():
        idx=0
        for result in results:
            for box in result.boxes:  # Assuming one result set
                class_id = int(box.cls)  # Get class ID (class label)
                class_name = result.names[class_id]  # Get class name from YOLO model
                box_minx, box_miny, box_maxx, box_maxy = map(int, box.xyxy[0])
                if class_name=='car' or class_name=='truck' or class_name=='bicycle':
                    if minx >= box_minx and miny >= box_miny and maxx <= box_maxx and maxy <= box_maxy and idx not in orient_set:
                        box_orient[idx] = box_to_pif(box_orient[idx],predictions[i])
                        orient_set.add(idx)
                        # Draw bounding box
                        rect = plt.Rectangle((box_minx, box_miny), box_maxx - box_minx, box_maxy - box_miny, linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                        
                        # Display orientation
                        ax.text(box_minx, box_miny - 5, f"Orient: {round(box_orient[idx], 2)}°", color='yellow', fontsize=12, backgroundcolor='black')
                        
                        ####visualization#####
                        
                        # cv2.rectangle(image_vis, (box_minx, box_miny), (box_maxx,box_maxy), (0, 255, 0), 2)
                        # # Compute center of the box
                        # c_x = (box_minx + box_maxx) // 2
                        # c_y = (box_miny + box_maxy) // 2
                    
                        # # Put the idx number at the center of the box
                        # cv2.putText(image_vis,str(idx),(c_x, c_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0),  2, cv2.LINE_AA)
                        # cv2.imshow("Object Detection with Centroids and Traffic Light Color", image_vis)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        break
                    
                    idx+=1       
    # # Save and show the output
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()

    ####show image###
    # # Show image using OpenCV
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display
    # cv2.imshow("Keypoints Visualization", img)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()

    for i in range (len(box_orient)):
        if i not in orient_set:
            box_orient[i]=-1*box_orient[i]


    print(f"Results saved to {output_path} and displayed.")
    
        

    return box_orient,orient_set
                  


class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_






# Directory containing input images
dir_path = "./SelfDrivingViz/Code/input_images/"

# Dynamically get all PNG images
image_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith((".png", ".jpg"))])
prev_image_path=None
alldata={}
bump_condition=False
bumpoff_count=0
prev_flow=0
for i,image_path in enumerate(image_paths):
    ###getting name of image ##
    optical_flow=None   
    if i==0 :
        prev_image_path=image_path
    else:
        optical_flow=raft_flow(prev_image_path,image_path)
        prev_image_path=image_path
        optical_flow=optical_flow.permute(0,2,3,1)
    #####################

    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    

    # Load YOLO model
    model = YOLOWorld("yolov8s-worldv2.pt")
    # Run object detection
    results = model(image_path)
    
    output_imagepath="./SelfDrivingViz/Code/2dout.png"
    detic_results=bridge_detic(image_path,output_imagepath)

    
    image = cv2.imread(image_path)
    image_check=image.copy()
    orig_image=image.copy()
    image_for_light=image.copy()
    # Get image dimensions
    height, width, _ = image.shape
    center_x = width // 2  # Image center along x-axis
    center_y=height //2
    #############zoe depth###############
    # Load depth estimation model
    pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-kitti")

    # Convert OpenCV image to PIL for depth estimation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use image_rgb for depth estimation
    image_pil = Image.fromarray(image_rgb)


    # Run depth estimation
    result = pipe(image_pil)
    depth_map = np.array(result["depth"])

    # Show the depth image
    depth_image = Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8))
    # depth_image.show() 

    #### Parameters of came ####
    fx = 1594.7
    fy = 1607.7
    ox = 655.30
    oy = 414.40

    


    #############getting height by adding contraints#######################
    z_w=[]
    z_t=[]

    bbox_list = []
    truckdict={}
    carrecord={}
    tempset=set()
    for result in results:
        count=0
        for box in result.boxes:
            class_id = int(box.cls)  # Get class ID (class label)
            class_name = result.names[class_id]
            if class_name=='car' or class_name=='truck' or class_name=='bicycle':
                
                if class_name=='bicycle':
                    class_name='cyclist'
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                #####marking the rectangle###
                # Draw bounding box (green) for all objects
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Compute center of the box
                c_x = (x1 + x2) // 2
                c_y = (y1 + y2) // 2
                
                if (c_x,c_y) in tempset :
                    continue
                elif (c_x,c_y) not in tempset:
                    tempset.add((c_x,c_y))


                if class_name=='truck' :
                    truckdict[count]=(c_x,c_y)
                elif class_name=='car':
                    carrecord[count]=(c_x,c_y)
                    
                # Put the idx number at the center of the box
                cv2.putText(image,str(count),(c_x, c_y),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0),  2, cv2.LINE_AA)

                ######################################
                # Extract the depth values within the bounding box
                depth_values = depth_map[y1:y2, x1:x2]
                count+=1
                top_left, bottom_right = (x1, y1), (x2, y2)
                bbox = [top_left, bottom_right]
                bbox_list.append(Bbox(bbox, class_name))
                # cv2.imshow("Keypoints Visualization", image)
                # cv2.waitKey(0)  # Wait for a key press to close the window
                # cv2.destroyAllWindows()
                #Calculate the average depth in the box
                if class_name=='car':
                    zdep = np.mean(depth_values)
                    ztemp=fy*1.44/np.abs(y2-y1)
                    z_w.append(zdep)
                    z_t.append(ztemp)
    
    for keytruck, value in truckdict.items():
        vx, vy = value
        min_dist = math.inf
        closest_car_key = None

        for keycar, point in carrecord.items():
            cx, cy = point
            dist = (cx - vx)**2 + (cy - vy)**2

            if dist < min_dist:
                min_dist = dist
                closest_car_key = keycar

        truckdict[keytruck] = closest_car_key  # Replace value with closest car index

    zscale_values = [ztemp / zw if zw != 0 else 0 for ztemp, zw in zip(z_t, z_w)]
    zscale = np.mean(zscale_values) if zscale_values else 0.35  # Avoid division by zero

    dets = bbox_list

    ####get box_orients ####
    dir_yolo3D = "/home/harmeet/Documents/Github/Computer_Vision/SelfDrivingViz/Code/YOLO3D/"

    # YOLO3D params
    weights = str(dir_yolo3D + 'yolov5s.pt')
    source = [image_path] # Provide your own image
    reg_weights = dir_yolo3D + 'weights/resnet18.pkl'
    model_select = 'resnet18'
    output_path = str(dir_yolo3D + 'runs')
    show_result = False
    save_result = False
    data = str(dir_yolo3D + 'data/coco128.yaml')
    imgsz = [640]
    device = ''
    classes = [0, 2, 3, 5] # Make sure this is correct !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    calib_file = str(dir_yolo3D + 'eval/camera_cal/calib_cam_to_cam.txt')




    box_orient=detect3d(
        dets,
        reg_weights,
        model_select,
        source,
        calib_file,
        show_result,
        save_result,
        output_path)
    
    # # Show the image with bounding boxes, centroids, and color annotations
    # cv2.imshow("Object Detection with Centroids and Traffic Light Color", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    orients = [math.degrees(angle) for angle in box_orient]
    
    box_orient,orient_set=run_inference(image_path,results,orients)
    print(box_orient)
    scaled_depthmap=zscale*depth_map
    #human_pose_list=human_pose(image_path,fx,fy,center_x,center_y,scaled_depthmap)
    lane_data=lane_info(image_path,i)
    
    alldata[image_name]={}
    alldata[image_name]["lanes"]=lane_data
    
    image_for_sign=cv2.imread(image_path)
    tempset.clear()
    #################this section gets the stationary objects ###########
    #####################remove outliers from it ###################
   














    def save_to_csv(results,detic_results,box_orient,orient_set, depth_map, idx):
        # Ensure the output directory exists
        global bump_condition
        global bumpoff_count
        global prev_flow
        output_dir = "./SelfDrivingViz/Code/outputs/"
        os.makedirs(output_dir, exist_ok=True)

        # Generate the file name dynamically
        file_name = os.path.join(output_dir, f"{idx}.csv")

        traffic_lights = []  # Store traffic lights with known statuses
        object_data = []  # Store all object data before writing

        # First pass: Process all objects
        nonmoving_set=('street_sign','trash_can','cone','barrel','telephone','traffic_light')
        nonmoving_dict={}
        for class_name, box in zip(detic_results["pred_class_names"],detic_results["boxes"]):
            x1, y1, x2, y2 = map(int, box)
            if class_name in nonmoving_set:
                nonmoving_dict[class_name]=(x1,y1,x2,y2)
           



        for class_name, box in zip(detic_results["pred_class_names"],detic_results["boxes"]):
            deticclasses=set()
            deticclasses=('street_sign','trash_can','telephone_pole','cone','barrel')
            if class_name not in deticclasses :
                continue
            obj_dict={}
            
            x1, y1, x2, y2 = map(int, box)  # Get box coordinates

            # Extract the depth values within the bounding box
            depth_values = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_values)
    
        # Compute the centroid of the bounding box
            centroid_x = (x1 + x2) // 2 
            centroid_y = (y1 + y2) // 2

            # Compute distance from image center
            xc = centroid_x - ox

            # Use the average depth (Z_w)
            zw = avg_depth

            # Calculate the angle and world coordinates
            hw = zw * zscale*0.8
            xw = (centroid_x - center_x) * hw / fx
            hw = hw*0.7
            #####saving nonmovingobjects in hashmap for opticalflow
            # Crop the image using the bounding box coordinates
            last_depth_check=[]

            # if len(prev_vehicle_record) and i>0:
            #     for cp,dp in prev_vehicle_record.items():
            #         if cp[0]-20<=centroid_x<=cp[0]+20 and cp[1]-20<=centroid_x<=cp[1]+20:
            #             last_depth_check.append(dp-(zw*zscale*0.8))
            # selfspeed=sum(x for x in last_depth_check)/len(last_depth_check)
            # prev_vehicle_record[(centroid_x,centroid_y)]=hw
            cropped_image = image_for_sign[y1:y2, x1:x2]


            cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            
            

            # If the detected class is 'street_sign', perform OCR
            if class_name == 'street_sign':

                sign_text_dict = checksign(cropped_image_pil)
                sign_text = sign_text_dict.get('<OCR>', '')
                print("sign_text",sign_text)


                lower_text = sign_text.lower()
                if "speedlimit" in lower_text  or '25' in lower_text or '20' in lower_text or '35'  in lower_text or '45' in lower_text :
                    import re
                    # Extract the first number from the text
                    match = re.search(r'\d+', sign_text)
                    if match:
                        speed_value = match.group()
                        label = f"speed={speed_value}"
                        obj_dict["SpeedSign"]=speed_value
                        cv2.rectangle(image_for_sign, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw centroid (small blue circle)
                        cv2.circle(image_for_sign, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                        cv2.putText(image_for_sign, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        obj_dict["Object Name"]=class_name
                        obj_dict["Xw (Sideways Distance)"]=xw
                        obj_dict["hw (Height)"]=hw
                        #obj_dict["Traffic Light Color"]=""
                        obj_dict["Rotation"]=0
                        # cv2.imshow("the signsbro", image_for_sign)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                if "speedhump" in lower_text:
                    bump_condition=True
                    bumpoff_count=0

            else:
                label = f"{class_name}"
                # Draw bounding box (green) for all objects
                #obj_dict["SpeedSign"]=""
                cv2.rectangle(image_for_sign, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw centroid (small blue circle)
                cv2.circle(image_for_sign, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                cv2.putText(image_for_sign, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                obj_dict["bbox"]=(x1,y1,x2,y2)
                obj_dict["Object Name"]=class_name
                obj_dict["Xw (Sideways Distance)"]=xw
                obj_dict["hw (Height)"]=hw
                #obj_dict["Traffic Light Color"]=""
                obj_dict["Rotation"]=0
            
                # cv2.imshow("the signsbro", image_for_sign)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            if len(obj_dict)>0:
                object_data.append(obj_dict)
            
            ####for speed bump #####

        if bump_condition:
            bump_path="./SelfDrivingViz/Code/lane.png"
            v=get_bump_loc(bump_path)
            if v==-1:
                bumpoff_count+=1
                if  bumpoff_count>6:
                    bumpoff_count=0
                    bump_condition=False
            else:
                xw=1
                if v == center_y:
                    zb=15
                else:
                    zb = fy * 0.9 / (v - oy)

                obj_dict["Object Name"]="bump"
                obj_dict["Xw (Sideways Distance)"]=xw
                obj_dict["hw (Height)"]=zb
                object_data.append(obj_dict)
        
        moving_threshold=5
        selfmoving=False
        if optical_flow is not None:
        
            ####get all the items of here from previous record################
            ####first get motion of self using stationary objects#########
            total_depth_flow=0
            for obj , box in nonmoving_dict.items():
                m1,n1,m2,n2=box
                mcx=m1+m2//2
                mcy=n1+n2//2
                nonmoving_crop_upper = optical_flow[0,n1:n1+10, (m1+m2)//2, :]
                nonmoving_crop_bottom = optical_flow[0,n2-10:n2, (m1+m2)//2, :]  
                depthflow_up = torch.mean(nonmoving_crop_upper[..., 1]).item()
                depthflow_bottom = torch.mean(nonmoving_crop_bottom[..., 1]).item()
                pxl_flow=np.abs(depthflow_up-depthflow_bottom)
                box_height=np.abs(n1-n2)
                mcx=m1+m2//2
                mcy=n1+n2//2
                px1=mcx-5
                px2=mcx+5
                py1=mcy-5
                py2=mcy+5
                depth_values = depth_map[n1:n2, m1:m2]
                zm = np.mean(depth_values)
                hm= zm*zscale*0.8
                # cv2.rectangle(image_check, (m1, n1), (m2, n2), (0, 255, 0), 2)
                # cv2.imshow("Object Detection with Centroids and Traffic Light Color", image_check)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #######applying velocity####always approaching
                vm=(hm * pxl_flow)/box_height
                total_depth_flow+=vm
            if len(nonmoving_dict)>0:
                net_world_depth_flow=total_depth_flow/len(nonmoving_dict)
                prev_flow=net_world_depth_flow
            else:
                net_world_depth_flow=prev_flow


            if net_world_depth_flow > 0.5:
                selfmoving=True


        feature_dict={}
        for class_name, box in zip(detic_results["pred_class_names"],detic_results["boxes"]):
              x1, y1, x2, y2 = map(int, box)
              if class_name=='reflector' or class_name=='headlight' or class_name=='taillight' or class_name=='hinge'or class_name=='pickup_truck' or class_name=='blinker' or class_name=='brake_light':
                feature_dict[(x1,y1,x2,y2)]=class_name
        
        vehicleset=('car','truck','bicycle','person','SUV','car_brakes_on')
       
        for result in results:
            count=0

            for index,box in enumerate(result.boxes):
                obj_dict={}
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                class_id = int(box.cls)  # Get class ID (class label)
                class_name = result.names[class_id]  # Get class name from YOLO model
                class_name = class_name.replace(" ", "_")
                if class_name=='bus':
                    continue
                


                
                
                # Extract the depth values within the bounding box
                depth_values = depth_map[y1:y2, x1:x2]
                avg_depth = np.mean(depth_values)

                # Compute the centroid of the bounding box
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                if (centroid_x,centroid_y) in tempset :
                    continue
                elif (centroid_x,centroid_y) not in tempset:
                    tempset.add((centroid_x,centroid_y))
                
                # Compute distance from image center
                xc = centroid_x - ox

                # Use the average depth (Z_w)
                zw = avg_depth

                # Calculate the angle and world coordinates
                hw = zw * zscale*0.8
                xw = (centroid_x - center_x) * hw / fx
                hw = hw*0.7
                # theta = math.atan((centroid_x - center_x) / fx)
                # hw = zw * zscale * math.cos(theta)*0.7
                # xw = zw * zscale * math.sin(theta)*0.7
                # cv2.rectangle(image_check, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.imshow("Object Detection with Centroids and Traffic Light Color", image_check)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # Default traffic light color
                traffic_light_color = ""
                speedsign=""
                # If the object is a traffic light, detect its color
                if "traffic_light" in class_name.lower():
                    hw=hw/0.8
                    # traffic_light_color = detect_traffic_light_color(image, [x1, y1, x2, y2])
                    # if traffic_light_color.lower() != "unknown":
                    #     traffic_lights.append((xw, hw, traffic_light_color, (x1, y1, x2, y2)))
                        #continue  # Skip writing traffic lights for now
                                        
                    chopped_image = image_check[y1:y2,x1:x2] # replace this with cropped bounding box
                    # cv2.imshow("Keypoints Visualization",chopped_image)
                    # cv2.waitKey(0)  # Wait for a key press to close the window
                    # cv2.destroyAllWindows()
                    color, shape, mask = identify_traffic_light(chopped_image, show=False) # Color: "red"/"yellow"/"green', Shape: "Circle"/"Arrow" 
                    print(f"Color: {color}, Shape: {shape}")
                    
                    if shape == "Arrow":
                        grey_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        direction = detect_arrow(grey_mask, show=False)
                        obj_dict["direction"]=direction

                    obj_dict["shape"]=shape
                    traffic_light_color=color

                aspect_ratio = np.abs(y2 - y1) / np.abs(x2 - x1)
                if "car" in class_name.lower()or 'bicycle' in class_name.lower():
                    rotation=box_orient[count]
                    count+=1
                elif    class_name=='truck':
                    if aspect_ratio < 0.50:
                            rotation=estimate_truck_orientation(image_check, x1, y1, x2, y2)
                            



                    else:
                        rotation=box_orient[truckdict[count]]
                        if ( -120<rotation<-70)or (70<rotation<120):
                            
                            dir=""
                            if feature_dict:
                                for key in feature_dict:
                                    xs1,ys1,xs2,ys2=key
                                    if x1<=xs1 and y1<=ys1 and x2>=xs2 and y2>=ys2:
                                        if feature_dict[key]=='hinge' or feature_dict[key]=='reflector':
                                            dir='back'
                                            break
                                        elif feature_dict[key]=='headlight':
                                            dir='front'
                                            break
                            if ( -120<rotation<-70 and dir=='back') or (70<rotation<120 and dir=='front'):
                                rotation=-1*rotation
                            count+=1
                    
                        
                else:
                    rotation=0
                # Store object data
                #### checking SUV 
                if "truck" in class_name.lower() :
                    if centroid_y<(0.70* height):
                        hw=hw+3
                    if centroid_y>(0.85* height):
                        xw=xw+1

                if class_name=='car':
                    h=zw * zscale * np.abs(y2-y1)/fy
                    if h>=1.65:
                        class_name='SUV'

                ####checking SUV########
                if class_name=='car':
                    if feature_dict:
                        for key in feature_dict:
                            xs1,ys1,xs2,ys2=key
                            cx=(xs1+xs2)//2
                            cy=(ys1+ys2)//2
                            if np.abs(centroid_x-cx)<5 and np.abs(centroid_y-cy)<5:
                                class_name='SUV'
                                break

                brake_light_check=False
                if class_name=='car' or class_name=='SUV':
                    backlightset=('taillight','brake_light','blinker')
                    if feature_dict:
                        for key in feature_dict:
                            if feature_dict[key] in backlightset:
                                xs1,ys1,xs2,ys2=key
                                cx=(xs1+xs2)//2
                                cy=(ys1+ys2)//2
                                if x1<=xs1 and y1<=ys1 and x2>=xs2 and y2>=ys2:
                                    brake_light_check=check_brakelight(image_check,key)
                    if brake_light_check:
                        class_name='car_brakes_on'          


                ####depth adjustment
                if class_name=='car' or   class_name=='SUV':
                    if centroid_y<(0.70* height):
                        hw=hw+1.2

                    if centroid_y>(0.85* height):
                        xw=xw+0.6

                ########################################optical flow################################
                ###################################################################################
                ###########moving and parking detection######################
                #if class_name in vehicleset and optical_flow is not None:
                if  optical_flow is not None and class_name in vehicleset:
                    # patchx1=centroid_x-10
                    # patchx2=centroid_x+10
                    # patchy1=centroid_y-10
                    # patchy2=centroid_y+10
                    # flow_crop = optical_flow[0, patchy1:patchy2, patchx1:patchx2, :]  # shape: [h, w, 2]
                    # widthflow = torch.mean(flow_crop[..., 0]).item()  # horizontal flow (u)
                    # depthflow = torch.mean(flow_crop[..., 1]).item()
                    # ####check if self_moving or not is moving or not#####
                    # depth_values=depth_map[ patchy1:patchy2, patchx1:patchx2]
                    # zopt = np.mean(depth_values)
                    # hopt= zopt*zscale*0.8
                    # delta_h=(hopt*(depthflow))/(centroid_y+depthflow-center_y)
                    # delta_h=np.abs(delta_h)
                    nonmoving_crop_upper = optical_flow[0,y1:y1+10, centroid_x, :]
                    nonmoving_crop_bottom = optical_flow[0,y2-10:y2, centroid_x, :]  
                    depthflow_up = torch.mean(nonmoving_crop_upper[..., 1]).item()
                    depthflow_bottom = torch.mean(nonmoving_crop_bottom[..., 1]).item()
                    pxl_flow=np.abs(depthflow_up-depthflow_bottom)
                    box_height=np.abs(y1-y2)
                    v_real=hw*pxl_flow/box_height
                    widththflow_up = torch.mean(nonmoving_crop_upper[..., 0]).item()
                    widthflow_bottom = torch.mean(nonmoving_crop_bottom[..., 0]).item()
                    xflow=np.abs((widthflow_bottom+widththflow_up)//2)
                    # if centroid_x>0.85*width and widthflow>0 and depthflow>0 and hw>12:
                    #     obj_dict["motion"]="parked"
                    # elif centroid_x<0.15*width and widthflow<0 and depthflow>0:
                    #     obj_dict["motion"]="parked"
                    # elif np.abs(delta_h)<np.abs(2.0*net_world_depth_flow):
                    #     obj_dict["motion"]="parked"
                    # else:
                    #     obj_dict["motion"]="moving"
                    if 0.60*net_world_depth_flow<v_real<1.50*net_world_depth_flow and hw<25:
                         class_name="car_parked"
                    # else:
                    #      class_name="car_moving"
                ####################collision detecton##############
                    if pxl_flow>0 :  ### means moving towards car
                        if centroid_x<0.5*width and xflow>0 or centroid_x>0.5*width and xflow<0:
                            if 5*v_real>=hw and hw<12:
                                #if (centroid_x <=center_x and widthflow>0) or (centroid_x >=center_x and widthflow<0) :
                                class_name="car_hitting"
                

                ##############################################################
                ########################################
                obj_dict["bbox"]=(x1,y1,x2,y2)
                obj_dict["Object Name"]=class_name
                obj_dict["Xw (Sideways Distance)"]=xw
                obj_dict["hw (Height)"]=hw
                obj_dict["Traffic Light Color"]=traffic_light_color
                obj_dict["Rotation"]=rotation
                #obj_dict["SpeedSign"]=speedsign
                object_data.append(obj_dict)

                # # Draw bounding box (green) for all objects
                

                # # Draw centroid (small blue circle)
                cv2.circle(image_check, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

                # # Put the class name label
                # label = f"{class_name}: {avg_depth:.2f}m"
                # #cv2.putText(image_check, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.imshow("Object Detection with Centroids and Traffic Light Color", image_check)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                

        return object_data


        # # Write data to CSV
        # with open(file_name, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Object Name", "Xw (Sideways Distance)", "Hw (Height)", "Traffic Light Color", "Rotation","SpeedSign"])
        #     writer.writerows(object_data)

    object_data=save_to_csv(results,detic_results,box_orient,orient_set, depth_map, i)
    alldata[image_name]["objects"]=object_data

    # # Show the image with bounding boxes, centroids, and color annotations
    # cv2.imshow("Object Detection with Centroids and Traffic Light Color", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




    # Adjust the plot so that Xw starts from the middle
    xw_list = []
    hw_list = []
    labels_list = []
    ###################kept seperate for plotting #################
    # Extract data for plotting
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            class_name = result.names[class_id]
            
            # Extract depth values and calculate average depth
            depth_values = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_values)
            
            # Compute centroid and distance from image center
            centroid_x = (x1 + x2) // 2
            xc = centroid_x - ox
            zw = avg_depth
            hw = zw * zscale*0.8
            xw = (centroid_x - center_x) * hw / fx
            hw=hw*0.7
            # theta = math.atan((centroid_x - center_x) / fx)
            # hw=zw*zscale*math.cos(theta)
            # xw=zw*zscale*math.sin(theta)
            # Store values for plotting
            xw_list.append(xw)
            hw_list.append(hw)
            labels_list.append(class_name)



    # Adjust the plot so that Xw starts from the middle
    # xw_centered = [xw - np.mean(xw_list) for xw in xw_list]
    xw_centered=xw_list
    # Plot world coordinates in Xw-Hw plane
    plt.figure(figsize=(8, 6))
    for idx in range(len(xw_list)):
        plt.scatter(xw_centered[idx], hw_list[idx], c='r', marker='o')
        plt.text(xw_centered[idx], hw_list[idx], labels_list[idx], fontsize=9, ha='right')
    import matplotlib.pyplot as plt
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("X_w (Sideways Distance)")
    plt.ylabel("H_w (Height)")
    plt.title("Top View of Object Positions (Xw-Hw Plane)")
    plt.grid(True)
    plt.savefig('./SelfDrivingViz/Code/finalpose.png')
   

# Specify output path for JSON
output_json_path = "./SelfDrivingViz/Code/output_data.json"

# Save alldata dictionary as a JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(alldata, json_file, indent=4)

print(f"Saved all image metadata to: {output_json_path}")
