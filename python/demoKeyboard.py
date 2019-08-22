from direct.showbase.ShowBase import ShowBase
from panda3d.core import (CollisionTraverser, CollisionHandlerEvent, DirectionalLight,
                          AmbientLight, VBase4, VBase3, BitMask32)
from OrbitCamera import CameraController
from CarK import KeyboardCar
from CarNN import NeuralNetworkCar
import math, csv, datetime

class SmartCar(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Override defaults
        self.disableMouse()
        self.setBackgroundColor(VBase3(160, 200, 150) / 255.0)
        self.setFrameRateMeter(True)

        # Lights
        dlight = DirectionalLight("dlight")
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(180.0, -70.0, 0)
        self.render.setLight(dlnp)

        alight = AmbientLight("alight")
        alnp = self.render.attachNewNode(alight)
        alight.setColor(VBase4(0.4, 0.4, 0.4, 1))
        self.render.setLight(alnp)

        # Collision traverser
        self.cTrav = CollisionTraverser("collisionTraverser")
        #self.cTrav.showCollisions(self.render)

        # Collision handlers
        self.carCollisionHandler = CollisionHandlerEvent()
        self.carCollisionHandler.addInPattern("%fn-into-%in")

        # Camera controls
        #self.cameraController = CameraController(self, 400, math.pi / 4.0, math.pi / 4.0)
        self.cameraController = CameraController(self, 300, -math.pi, math.pi / 4.0)

        # Load the track
        self.track = self.loader.loadModel("models/trackMotegi")
        #self.track = self.loader.loadModel("models/trackValencia")
        checkpointsCollision = self.track.find("checkpoints").node()
        checkpointsCollision.setIntoCollideMask(BitMask32(0xF0))
        self.numCheckpoints = checkpointsCollision.getNumSolids()
        self.track.reparentTo(self.render)

        # Load the car
        self.car = KeyboardCar(self)
        #self.car = NeuralNetworkCar(self)
        #self.cameraController.follow(self.car.getNodePath())
        self.cameraController.setTarget(self.car.getNodePath())

        # Reposition the car
        #self.car.getNodePath().setH(180.0)

        # Register car collisions with track
        self.cTrav.addCollider(self.car.carCollider, self.carCollisionHandler)
        self.accept("carCollider-into-trackCollision", self.car.onCrash)
        self.accept("carCollider-into-checkpoints", self.car.onCheckpoint)

        # State logger
        self.loggingActive = False
        self.log = []
        self.accept("l", self.toggleStateLogger)

    def toggleStateLogger(self):
        if not self.loggingActive:
            print "LOG: started."
            self.taskMgr.add(self.stateLogger, "StateLoggerTask")
        else:
            self.taskMgr.remove("StateLoggerTask")
            self.writeLog("data/log.csv")
        self.loggingActive = not self.loggingActive

    def stateLogger(self, task):
        row = [self.car.speed, self.car.steerAngle]
        row.extend(self.car.sensorDistances)
        row.append(float(self.car.arrowUpDown))
        row.append(float(self.car.arrowLeftDown))
        row.append(float(self.car.arrowRightDown))
        self.log.append(row)
        return task.again

    def writeLog(self, filename):
        with open(filename, 'wb') as file:
            wr = csv.writer(file)
            wr.writerows(self.log)
        print "LOG: written."

main = SmartCar()
desiredFPS = 30.0
delay = 1.0 / desiredFPS
startTime = datetime.datetime.now()
while True:
    currentTime = datetime.datetime.now()
    diffTime = currentTime - startTime
    diff = diffTime.total_seconds()
    if diff > delay:
        startTime = currentTime
        main.taskMgr.step()
