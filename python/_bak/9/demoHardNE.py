from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (CollisionTraverser, CollisionHandlerEvent, DirectionalLight,
                          AmbientLight, VBase4, VBase3, BitMask32, TextNode, LColor)
from OrbitCamera import CameraController
from CarNN import NeuralNetworkCar
from CarNE import NeuroevolutionCar
from CarK import KeyboardCar
import math, datetime
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
        self.cameraController = CameraController(self, 600, math.radians(45), math.radians(60))

        # Load the track
        self.track = self.loader.loadModel("models/trackMotegi")
        # self.track = self.loader.loadModel("models/trackValencia")
        checkpointsCollision = self.track.find("checkpoints").node()
        checkpointsCollision.setIntoCollideMask(BitMask32(0xF0))
        self.numCheckpoints = checkpointsCollision.getNumSolids()
        self.track.reparentTo(self.render)

        # Enemy cars
        enemyColor = LColor(204 / 255.0, 72 / 255.0, 63 / 255.0, 1.0)
        self.enemyCars = []
        car = NeuralNetworkCar(self)
        car.getNodePath().setX(-150)
        car.getNodePath().setY(-110)
        car.getNodePath().setH(60)
        car.getNodePath().setColor(enemyColor)
        self.enemyCars.append(car)

        car = NeuralNetworkCar(self)
        car.getNodePath().setX(113)
        car.getNodePath().setY(-233)
        car.getNodePath().setH(45)
        car.getNodePath().setColor(enemyColor)
        self.enemyCars.append(car)

        car = NeuralNetworkCar(self)
        car.getNodePath().setX(-255)
        car.getNodePath().setY(-264)
        car.getNodePath().setH(20)
        car.getNodePath().setColor(enemyColor)
        self.enemyCars.append(car)

        car = NeuralNetworkCar(self)
        car.getNodePath().setX(-364)
        car.getNodePath().setY(-237)
        car.getNodePath().setH(-90)
        car.getNodePath().setColor(enemyColor)
        self.enemyCars.append(car)

        car = NeuralNetworkCar(self)
        car.getNodePath().setX(-126)
        car.getNodePath().setY(-5)
        car.getNodePath().setH(180)
        car.getNodePath().setColor(enemyColor)
        self.enemyCars.append(car)

        # Neuroevolution
        self.inputLayerSize = 13
        self.hiddenLayer1Size = 10
        self.hiddenLayer2Size = 5
        self.numLabels = 2
        self.thetaSizes = [(self.hiddenLayer1Size, self.inputLayerSize + 1),
                           (self.hiddenLayer2Size, self.hiddenLayer1Size + 1),
                           (self.numLabels, self.hiddenLayer2Size + 1)]

        self.generationSize = 20
        self.weightInit = 0.12
        self.replaceRatio = 0.05
        self.scaleRatio = 0.05
        self.addRatio = 0.05

        self.generationCount = 1
        self.cars = []
        for i in range(self.generationSize):
            car = NeuroevolutionCar(self, i, self.inputLayerSize)
            # Register car collisions with track
            self.cTrav.addCollider(car.carCollider, self.carCollisionHandler)
            self.accept("carCollider{}-into-trackCollision".format(i), car.onCrash)
            self.accept("carCollider{}-into-checkpoints".format(i), car.onCheckpoint)
            self.cars.append(car)

        # Initial generation
        np.random.seed(1)
        for i in range(self.generationSize):
            theta1 = self.randWeights(self.thetaSizes[0], self.weightInit)
            theta2 = self.randWeights(self.thetaSizes[1], self.weightInit)
            theta3 = self.randWeights(self.thetaSizes[2], self.weightInit)
            self.cars[i].startSimulation(theta1, theta2, theta3)

        # Run learning task
        self.taskMgr.add(self.neuroevolution, "NeuroevolutionTask")

        # DEBUG
        self.txtGen = OnscreenText(text='', pos=(0.0, -0.04), scale=0.05,
                                   align=TextNode.ALeft, fg=(1, 1, 1, 1), bg=(0, 0, 0, .4))
        self.txtGen.reparentTo(self.a2dTopLeft)

    def neuroevolution(self, task):
        generationDone = True
        for car in self.cars:
            if not car.simulationDone:
                generationDone = False

        if generationDone:
            # Compute fitness scores
            fitness = np.zeros(self.generationSize)
            for i in range(self.generationSize):
                fitness[i] = self.fitnessFunction(self.cars[i].checkpointCount, self.cars[i].timeAlive)
            prob = fitness / np.sum(fitness)

            # Generate new generation
            nextGenTheta1 = []
            nextGenTheta2 = []
            nextGenTheta3 = []
            for i in range(self.generationSize):
                # Selection
                (p1_i, p2_i) = np.random.choice(self.generationSize, 2, replace=False, p=prob)
                if fitness[p1_i] < fitness[p2_i]:
                    p1_i, p2_i = p2_i, p1_i

                r1, c1 = self.cars[p1_i].theta1.shape
                r2, c2 = self.cars[p1_i].theta2.shape
                r3, c3 = self.cars[p1_i].theta3.shape

                # Crossover
                fit1 = fitness[p1_i]
                fit2 = fitness[p2_i]
                crossoverRatio = fit2 / (fit1 + fit2)

                theta1 = np.copy(self.cars[p1_i].theta1)
                mask1 = np.random.rand(r1, c1) < crossoverRatio
                theta1[mask1] = self.cars[p2_i].theta1[mask1]

                theta2 = np.copy(self.cars[p1_i].theta2)
                mask2 = np.random.rand(r2, c2) < crossoverRatio
                theta2[mask2] = self.cars[p2_i].theta2[mask2]

                theta3 = np.copy(self.cars[p1_i].theta3)
                mask3 = np.random.rand(r3, c3) < crossoverRatio
                theta3[mask3] = self.cars[p2_i].theta3[mask3]

                # Mutation
                # replace
                maskReplace = np.random.rand(r1, c1) < self.replaceRatio
                sz = np.sum(maskReplace)
                theta1[maskReplace] = np.random.rand(sz) * 2 * self.weightInit - self.weightInit

                maskReplace = np.random.rand(r2, c2) < self.replaceRatio
                sz = np.sum(maskReplace)
                theta2[maskReplace] = np.random.rand(sz) * 2 * self.weightInit - self.weightInit

                maskReplace = np.random.rand(r3, c3) < self.replaceRatio
                sz = np.sum(maskReplace)
                theta3[maskReplace] = np.random.rand(sz) * 2 * self.weightInit - self.weightInit

                # scale
                maskScale = np.random.rand(r1, c1) < self.scaleRatio
                sz = np.sum(maskScale)
                theta1[maskScale] = theta1[maskScale] * (np.random.rand(sz) + 0.5)

                maskScale = np.random.rand(r2, c2) < self.scaleRatio
                sz = np.sum(maskScale)
                theta2[maskScale] = theta2[maskScale] * (np.random.rand(sz) + 0.5)

                maskScale = np.random.rand(r3, c3) < self.scaleRatio
                sz = np.sum(maskScale)
                theta3[maskScale] = theta3[maskScale] * (np.random.rand(sz) + 0.5)

                # add
                maskAdd = np.random.rand(r1, c1) < self.addRatio
                sz = np.sum(maskAdd)
                theta1[maskAdd] = theta1[maskAdd] + (np.random.rand(sz) - 0.5) * 2.0

                maskAdd = np.random.rand(r2, c2) < self.addRatio
                sz = np.sum(maskAdd)
                theta2[maskAdd] = theta2[maskAdd] + (np.random.rand(sz) - 0.5) * 2.0

                maskAdd = np.random.rand(r3, c3) < self.addRatio
                sz = np.sum(maskAdd)
                theta3[maskAdd] = theta3[maskAdd] + (np.random.rand(sz) - 0.5) * 2.0

                nextGenTheta1.append(theta1)
                nextGenTheta2.append(theta2)
                nextGenTheta3.append(theta3)

            # Run new generation
            for i in range(self.generationSize):
                self.cars[i].startSimulation(nextGenTheta1[i], nextGenTheta2[i], nextGenTheta3[i])

            self.generationCount += 1
            self.txtGen.setText('Gen: {}'.format(self.generationCount))

        return task.cont

    def fitnessFunction(self, checkpointCount, timeAlive):
        f = 100 * checkpointCount + timeAlive + 1
        return f

    def randWeights(self, size, weightInit):
        w = np.random.rand(size[0], size[1]) * 2 * weightInit - weightInit
        return w

main = LearnNeuroevolution()
desiredFPS = 200.0
delay = 1.0 / desiredFPS
startTime = datetime.datetime.now()
while True:
    currentTime = datetime.datetime.now()
    diffTime = currentTime - startTime
    diff = diffTime.total_seconds()
    if diff > delay:
        startTime = currentTime
        main.taskMgr.step()
