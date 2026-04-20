class V2RCorrdinates:
    def __init__(self, u, v):
        self.u = u
        self.v = v
    
    def calculation(self, Z):

        K = [[1440, 0 , 960], [0, 1440, 540], [0, 0, 1]]

        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        t = 3.66

        a = Z/(math.cos(math.pi/12) * 2)
        Z1 = Z*math.sqrt(2) - a
        theta = math.atan(t/Z1)
        b = a * math.tan(theta)
        c = Z*math.sqrt(2)*math.tan(theta)/(math.cos(theta) - math.sin(theta))
        e = t + a * math.tan(theta)

        y = math.sqrt(e**2 + c**2 - 2*e*c*math.cos(math.pi/2 + theta))

        forwardFinal = y + Z
        return forwardFinal
