import rclpy
import rospy 
from std_msgs.msg import String
from math import asin as asin
from math import atan as atan
from math import sqrt as sqrt
from trajectory_msgs.msg import motorSpeeds
from trajectory_msgs.msg import JointTrajectoryPoint
#values to send to the robot is in m/s starts at 0.5
r = 0.12 #12 cm
l = 0.46 #46 cm
#dimensions for image 640x480
#assumptions for this file will be given perpendicular distance
#and angle of the edges of the side walk

	
global messageData

def callback(msg):
	messageData = msg


threshold = 1

#pull in the coordinates from the camera 
sub = rospy.Subscriber('/camera_line_processor', Int32,callback)
pub = rospy.Publisher('/motor_speeds',motorSpeeds)
rospy.init_node('/getPath', anonymous=True)
xgoal, ygoal, screenResx,screenResy = messageData[0],messageData[1],messageData[2],message[3]

orientation = 0



#given the line use proportional control along for a DD drive robot
hystoresis = 1
wheelV = 3
desiredTheta = 0
currentx,currenty = screenResx/2,0

while (len(messageData[0]) != 0): 

	
	#x = x - (screenRes/2) 

	if xgoal < 290:
		pub.publish(1.25,1.25,1,1)
	if xgoal > 330:
		pub.publish(1,1,1.25,1.25)
	if (xgoal < 330 and xgoal > 290)
		pub.publish(1,1,1,1)

	#th_err = atan2(ygoal-currenty,xgoal-currentx) - orientation
	#d1 = abs(currentx - xgoal)
	#d2 = abs(currenty - ygoal)
	#w = 1
	#d = sqrt(d1*d1+d2*d2)
	#if(d < hystoresis)
	#	pub.publish(0,0,0,0)
	#	break
	#w1 = w + wheelV*th_err
	#w2 = w - wheelV*th_err
	#if(d < 2)
	#	w1,w2 = wheelV*d*(w+wheelV*th_err),wheelV*d*(w-wheelV*th_err)
	
	
	pub.publish(w1,w1,w2,w2)

	#dth = (r/(2.0*l))*(w1-w2)
	#orientation = dth + orientation
	
	




























