from styx_msgs.msg import TrafficLight
import cv2
import os
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.state = 0
        self.img_ctr = 0
        self.working_directory = os.path.dirname(os.path.realpath(__file__))
        
        # traffic light detection
        self.detect_tl = tf.Graph()
        
        #traffic light color classification
        self.classify = tf.Graph()      #computation graph definition
        
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
                
                cv2.imwrite(self.working_directory + "/traffic_sign_images/Test_" + str(self.img_ctr) + ".jpg", box) 
                rospy.logerr('Image counter = %d', self.img_ctr)

            # Add classification code here


            return TrafficLight.UNKNOWN
    
    def locate_lights (self, image):
          # Actual detection.
        #switch from BGR to RGB
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        tf_input_image = np.expand_dims(image, axis=0)

        
        with self.detect_tl.as_default():
            (boxes, scores, classes, num_detections) = self.detect_session.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                feed_dict={self.image_tensor: tf_input_image})

            boxes=np.squeeze(boxes)         # remove single dimension entries
            #rospy.logwarn('shape of scores BEFORE squeeze = ')
            #rospy.logwarn(scores.shape)
            #rospy.logwarn('shape of Boxes BEFORE squeeze = ')#, boxes.shape)
            #rospy.logwarn(boxes.shape)
            #scores=np.squeeze(scores)
            classes=np.squeeze(classes)#.astype(np.int32)
            num_detections = np.squeeze(num_detections)
            
            #rospy.logwarn('shape of scores AFTER squeeze = ')
            #rospy.logwarn(scores.shape)
            rospy.logwarn('shape of Boxes AFTER squeeze = ')#, boxes.shape)
            rospy.logwarn(boxes.shape)
            
            tl_box_image = None
            detection_score = 0.5
            
            for i,image_class in enumerate(classes.tolist()):
                if image_class == 10:# and socres[i] >=detection_score:               # 10 = "traffic light" 
                        rospy.logwarn('Traffic image found!')#, boxes.shape)
                        return boxes[i]
                else:
                    pass
                
        return -1
            
