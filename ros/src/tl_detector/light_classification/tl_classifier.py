from styx_msgs.msg import TrafficLight
import cv2
import os
import tensorflow as tf
import numpy as np
import rospy
import keras
from keras.models import load_model

#from utils import label_map_util
#from utils import visualization_utils as vis_util

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        self.state = 0
        self.img_ctr = 0
        self.working_directory = os.path.dirname(os.path.realpath(__file__))
        keras.backend.clear_session()

        
        # traffic light detection
        self.detect_tl = tf.Graph()
        
        #traffic light color classification
        self.classify = tf.Graph()      #computation graph definition
        with self.classify.as_default():
           
            self.model = load_model(self.working_directory + '/models/tl_classifier_trained.h5')
            self.model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        
        with self.detect_tl.as_default():
            graph_def = tf.GraphDef()                                       # to read pb file (the detection model) protobuf file
            
            with open(self.working_directory + "/models/frozen_inference_graph.pb", 'rb') as f:    # rb means reading binary
                graph_def.ParseFromString(f.read())
                tf.import_graph_def( graph_def, name="" )
            
            self.detect_session = tf.Session(graph=self.detect_tl)
            # Image tensor Accepts a uint8 4-D tensor of shape [None, None, None, 3]
            self.image_tensor = self.detect_tl.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detect_tl.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.detect_tl.get_tensor_by_name('detection_scores:0')
            # Class represent the ID number of the tensor detected
            self.classes = self.detect_tl.get_tensor_by_name('detection_classes:0') # traffic light = 10
            # Number of valid boxes per image
            self.num_detections = self.detect_tl.get_tensor_by_name('num_detections:0')
  # Actual detection.



    def get_classification(self, image):
            """Determines the color of the traffic light in the image
            Args:
                image (cv::Mat): image containing the traffic light
            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            """
            #TODO implement light color prediction
            box = self.locate_lights(image)
            if box is not None:
                self.img_ctr +=1
                #rospy.logwarn('box dimensions')
                #rospy.logwarn(box)
                #rospy.logwarn(box[1])
                #rospy.logwarn(box[2])
                #rospy.logwarn(box[3])
                sliced_image = cv2.resize( image[box[0]:box[2], box[1]:box[3]], (32,32) )
                cv2.imwrite(self.working_directory + "/traffic_sign_images/Test_" + str(self.img_ctr) + ".jpg", sliced_image) 
                rospy.logerr('Image counter = %d', self.img_ctr)
                sliced_image = np.reshape(sliced_image,[1,32,32,3])
                with self.classify.as_default():
                    rospy.logwarn('im in the prediction business')
                    classes = self.model.predict_classes(sliced_image)[0]
                    rospy.logwarn(classes)
                    if classes == 1:
                        return TrafficLight.RED 

            # Add classification code here

                    else:
                        return TrafficLight.UNKNOWN
    
    def locate_lights (self, image):
          # Actual detection.
        #switch from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        tf_input_image = np.expand_dims(image, axis=0)

        
        with self.detect_tl.as_default():
            (boxes, scores, classes, num_detections) = self.detect_session.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                feed_dict={self.image_tensor: tf_input_image})

            boxes=np.squeeze(boxes)         # remove single dimension entries
            #scores=np.squeeze(scores)
            #rospy.logwarn('scores ')
            #rospy.logwarn(scores)
            classes=np.squeeze(classes)#.astype(np.int32)
            #rospy.logwarn('classes ')
            #rospy.logwarn(classes)
            #rospy.logwarn('sliced IMAGE shape ')
            #rospy.logwarn(image.shape)
            img_h = image.shape[0]
            img_w = image.shape[1]
            
            #num_detections = np.squeeze(num_detections)

            #rospy.logwarn('shape of Boxes AFTER squeeze = ')#, boxes.shape)
            #rospy.logwarn(boxes.shape)

        #return image
            #vis_util.visualize_boxes_and_labels_on_image_array(
            #    image, # tf_input_image
            #   np.squeeze(boxes),
            #    np.squeeze(classes).astype(np.int32),
            #    np.squeeze(scores),
            #    category_index,
            #    use_normalized_coordinates=True,
            #    line_thickness=8,
            #    min_score_thresh=0.80)
          
        
        
        #    tl_box_image = None
            detection_score = 0.5
            
            for i,image_class in enumerate(classes.tolist()):
                dummy = 0
                if image_class == 10:
                        #rospy.logwarn(scores[0][i])
                        dummy+= scores[0][i]
                        if dummy >= detection_score:               # 10 = "traffic light" 
                            #rospy.logwarn('Traffic image found!')#, boxes.shape)
                            #rospy.logwarn(dummy)
                            traffic_light_img_box = self.slice_image(boxes[i],img_h,img_w)
                            dummy = 0
                            return traffic_light_img_box #boxes[i]
                        
                else:
                    pass
                
        return None
    
    def slice_image(self, box_dim, img_h, img_w):
        #defining sliced image corners
        
        
        c1 = box_dim[0]*img_h
        c2 = box_dim[1]*img_w
        c3 = box_dim[2]*img_h
        c4 = box_dim[3]*img_w
        box = [int(c1), int(c2), int(c3), int(c4)]
        return np.array(box)
