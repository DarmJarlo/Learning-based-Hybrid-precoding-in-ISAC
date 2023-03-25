class RSU:
    def __init__(self):
        self.DistanceToLane = 500
        self.DistanceToVehicle = []
        self.precoding_matrix = np.zeros([8,8])#precoding_matrix : Nt*K
        self.location = (0,0)

