import math
import numpy as np

###### FoV horizontal and vertical both are 30 degrees.
class V2RGroundQRRed:
    def __init__(self, u, v, Z):
        self.u = u
        self.v = v
        self.Z = Z
        self.forwardFinal =self.forwardFinal
    
    def calculation(self, u, v, Z): # doing something with the inputs and storing it.

        K = [[1440, 0 , 960], [0, 1440, 540], [0, 0, 1]]

        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        # t = 3.66
        s = (u - cx) * Z1 / fx
        t = (v - cy) * Z1 / fy

        # For x / sideways transform.
        x = s*Z*math.sqrt(2)/Z1

        theta = math.atan(t/Z1)

        if t >= 0:
            # For y / forward transformation.
            a = Z/(math.cos(math.pi/12) * 2)
            Z1 = Z*math.sqrt(2) - a
            b = a * math.tan(theta)
            c = Z*math.sqrt(2)*math.tan(theta)/(math.cos(theta) - math.sin(theta))
            e = t + a * math.tan(theta)
            y = np.sign(v - cy) * math.sqrt(e**2 + c**2 - 2*e*c*math.cos(math.pi/2 + theta))

            forwardFinal = y + Z
            relative_coordinates = [x, forwardFinal, Z]

            return relative_coordinates
        
        else:
            c_alt = Z * math.sqrt(2) - Z/math.cos(math.pi/4 + theta) * math.cos(theta)
            # a_alt = a - c_alt
            # b_alt = a * math.tan(theta)
            # e_alt = t - b_alt
            # y_negForm = np.sign(v - cy) * math.sqrt(e_alt**2 + c_alt**2 - 2*e_alt*c_alt*math.cos(math.pi/2 + theta))

            forwardFinalAlt = -y + Z
            relative_coordinatesAlt = [x, forwardFinalAlt, Z]
            return relative_coordinatesAlt
        



class V2RVerticalBanner:
    def __init__(self, u, v, Z, lidarStraight):
        self.u = u
        self.v = v
        self.Z = Z
        self.lidarStraight = lidarStraight

    def calculation(self, u, v, Z, lidarStraight0deg):

        K = [[1440, 0 , 960], [0, 1440, 540], [0, 0, 1]]

        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        # t = 3.66
        s = (u - cx) * l1 / fx
        t = (v - cy) * l1 / fy

        l = lidarStraight0deg
        theta = math.atan(t/l1)
        x = s*l*math.sqrt(2)/l1

        if t < 0:
            l1 = l/(math.cos(math.pi/12) * math.cos(math.pi/6))
            a = l * math.sqrt(2) - l1
            b = a * math.tan(theta) #b is neg.
            # l1 = l*math.sqrt(2) - a
            e = t-b # negative
            c = l/math.cos(math.pi/4 - theta) - l * math.sqrt(2)/math.cos(theta)

            z = l * math.sqrt(2) + math.sqrt(e**2 + c**2 - 2*(-e)*c*math.cos(math.pi/2 - theta))

            finalCoordV = [x , l, z]
            return finalCoordV
        
        else:
            z_alt = l * math.sqrt(2) - l * math.cos(theta) / math.cos(math.pi/4 - theta)

            finalCoordV_alt = [x, l, z_alt]
            return finalCoordV_alt
