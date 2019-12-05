import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

global messageData

def callback(msg):
	messageData = msg


pub = rospy.Publisher('/cmd_joint_traj', JointTrajectory, queue_size=10)
rospy.init_node('/run_single_motor_node', anonymous=True)
sub = rospy.sub('/motor_speeds', Int32,callback)
rate = rospy.Rate(10) # 10hz

while 1: 

	lb,lf,rb,rf = messageData[0], messageData[1],messageData[2],messageData[3]
	point = JointTrajectoryPoint()
	point.velocities = [lb,lf,rb,rf]
	message = JointTrajectory()
	message.joint_names = ['wheel_back_left_joint', 'wheel_front_left_joint', 'wheel_back_right_joint','wheel_front_right_joint']
	message.points = [point]
        pub.publish(message)
        rate.sleep()
	


