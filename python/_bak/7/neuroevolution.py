from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (CollisionTraverser, CollisionHandlerEvent, DirectionalLight,
                          AmbientLight, VBase4, VBase3, BitMask32, TextNode)
from OrbitCamera import CameraController
from CarNE import NeuroevolutionCar
import math, csv, datetime
import numpy as np


class LearnNeuroevolution(ShowBase):
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

        # Collision handlers
        self.carCollisionHandler = CollisionHandlerEvent()
        self.carCollisionHandler.addInPattern("%fn-into-%in")

        # Camera controls
        self.cameraController = CameraController(self, 400, math.pi / 4.0, math.pi / 4.0)

        # Load the track
        self.track = self.loader.loadModel("models/trackMotegi")
        #self.track = self.loader.loadModel("models/trackValencia")
        checkpointsCollision = self.track.find("checkpoints").node()
        checkpointsCollision.setIntoCollideMask(BitMask32(0xF0))
        self.numCheckpoints = checkpointsCollision.getNumSolids()
        self.track.reparentTo(self.render)

        # Neuroevolution
        self.generationCount = 0
        self.generationSize = 20
        self.cars = []
        for i in range(self.generationSize):
            car = NeuroevolutionCar(self, i)
            # Register car collisions with track
            self.cTrav.addCollider(car.carCollider, self.carCollisionHandler)
            self.accept("carCollider{}-into-trackCollision".format(i), car.onCrash)
            self.accept("carCollider{}-into-checkpoints".format(i), car.onCheckpoint)
            self.cars.append(car)

        # Run learning task
        self.taskMgr.add(self.neuroevolution, "NeuroevolutionTask")

        # DEBUG
        self.txtGen = OnscreenText(text='', pos=(0.0, -0.04), scale=0.05, align=TextNode.ALeft, fg=(1, 1, 1, 1), bg=(0, 0, 0, .4))
        self.txtGen.reparentTo(self.a2dTopLeft)

    def neuroevolution(self, task):
        generationDone = True
        for car in self.cars:
            if not car.simulationDone:
                generationDone = False

        if generationDone:
            self.generationCount += 1
            self.txtGen.setText('Gen: {}'.format(self.generationCount))

            # Compute fitness functions

            for i in range(self.generationSize):
                theta1 = self.randWeights(5, 10)
                theta2 = self.randWeights(5, 6)
                theta3 = self.randWeights(2, 6)
                self.cars[i].startSimulation(theta1, theta2, theta3)

        return task.cont

    def randWeights(self, rows, cols):
        epsilonInit = 0.12
        w = np.random.rand(rows, cols) * 2 * epsilonInit - epsilonInit
        return w

main = LearnNeuroevolution()
desiredFPS = 60.0
delay = 1.0 / desiredFPS
startTime = datetime.datetime.now()
while True:
    currentTime = datetime.datetime.now()
    diffTime = currentTime - startTime
    diff = diffTime.total_seconds()
    if diff > delay:
        startTime = currentTime
        main.taskMgr.step()
