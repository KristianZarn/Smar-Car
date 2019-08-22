from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, DirectionalLight, AmbientLight, LineSegs, Point3, VBase4
from OrbitCamera import CameraController
from Car import KeyboardController
import math

class SmartCar(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Override defaults
        self.disableMouse()
        self.setBackgroundColor(0.7, 0.7, 0.7)
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

        # Collisions
        self.cTrav = CollisionTraverser("collisionTraverser")
        self.cTrav.showCollisions(self.render)

        # Camera controls
        self.cameraController = CameraController(self, 200, math.pi / 4.0, math.pi / 4.0)
        #self.cameraController = CameraController(self, 200, -math.pi, math.pi / 4.0)

        # Load the track
        self.track = self.loader.loadModel("models/trackMotegi")
        self.track.reparentTo(self.render)

        # Load the car
        self.car = KeyboardController(self)
        #self.cameraController.setTarget(self.car.getNodePath())

main = SmartCar()
main.run()