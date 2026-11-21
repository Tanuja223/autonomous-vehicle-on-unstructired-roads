"""simulation/carla_env.py - GPU-safe CARLA environment."""



from __future__ import annotations



import math

import random

import time

from dataclasses import dataclass

from enum import Enum, auto

from typing import Any, Dict, List, Optional, Tuple



import numpy as np



try:

    import gymnasium as gym

    from gymnasium import spaces

    GYM_AVAILABLE = True

except ImportError:

    GYM_AVAILABLE = False



try:

    import carla

    CARLA_AVAILABLE = True

except ImportError:

    CARLA_AVAILABLE = False

    print("[CarlaEnv] carla module not found – running in mock mode.")





# UNSTRUCTURED ROAD MODE
# OSM autonomous driving can use world-space local targets generated in
# main.py instead of CARLA LaneType.Driving waypoints or the A* road graph.
#
# The tested 64x48 RGB camera, asynchronous CARLA mode, and delayed camera
# startup are intentionally preserved for the RTX 2050 + CARLA 0.9.15 setup.
#
# NOTE: Town10HD remains the visual/physics map. "Unstructured" here means
# the driving policy does not follow lane markings or the CARLA road graph.
# A dirt/rock terrain visual layer can be added separately later.
class Scenario(Enum):

    STRAIGHT_ROAD = auto()

    INTERSECTION = auto()

    PEDESTRIAN_CROSS = auto()

    DENSE_URBAN = auto()

    NIGHT_DRIVING = auto()





SCENARIO_CONFIGS: Dict[Scenario, dict] = {

    Scenario.STRAIGHT_ROAD: {"npcs": 5, "peds": 0, "weather": "ClearNoon", "map": "Town04"},

    Scenario.INTERSECTION: {"npcs": 15, "peds": 5, "weather": "ClearNoon", "map": "Town03"},

    Scenario.PEDESTRIAN_CROSS: {"npcs": 5, "peds": 20, "weather": "CloudyNoon", "map": "Town01"},

    Scenario.DENSE_URBAN: {"npcs": 50, "peds": 30, "weather": "ClearNoon", "map": "Town03"},

    Scenario.NIGHT_DRIVING: {"npcs": 10, "peds": 5, "weather": "ClearNight", "map": "Town03"},

}





@dataclass

class SensorBundle:

    rgb_image: Optional[np.ndarray] = None

    depth_image: Optional[np.ndarray] = None

    semantic_seg: Optional[np.ndarray] = None

    lidar_points: Optional[np.ndarray] = None

    collision_detected: bool = False

    lane_invaded: bool = False





BASE_CLASS = (gym.Env if GYM_AVAILABLE else object)





class CarlaEnv(BASE_CLASS):

    """Lightweight Gymnasium-compatible CARLA environment.



    Important GPU-safe design:

    - CARLA RGB camera renders at only 64x48.

    - The frame is resized on the CPU to the existing 288x160 API size.

    - CARLA runs asynchronously.

    - No Traffic Manager/NPC startup.

    - No manual world.tick() in async mode.

    """



    # Keep the project's original observation API.

    IMG_H = 160

    IMG_W = 288

    MAX_EPISODE_STEPS = 3000



    def __init__(

        self,

        host: str = "localhost",

        port: int = 2000,

        timeout: float = 30.0,

        scenario: Scenario = Scenario.STRAIGHT_ROAD,

        synchronous: bool = False,

        delta_seconds: float = 0.05,

        render_mode: Optional[str] = None,

    ) -> None:

        super().__init__()



        self.host = host

        self.port = port

        self.timeout = timeout

        self.scenario = scenario

        self.synchronous = synchronous

        self.delta_t = delta_seconds

        self.render_mode = render_mode



        if GYM_AVAILABLE:

            self.observation_space = spaces.Dict({

                "camera": spaces.Box(

                    0, 255, (self.IMG_H, self.IMG_W, 3), dtype=np.uint8

                ),

                "speed": spaces.Box(-1, 200, (1,), dtype=np.float32),

                "steer": spaces.Box(-1, 1, (1,), dtype=np.float32),

                "lane_pos": spaces.Box(-5, 5, (1,), dtype=np.float32),

            })

            self.action_space = spaces.Box(

                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),

                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),

            )



        self._client: Optional[Any] = None

        self._world: Optional[Any] = None

        self._ego: Optional[Any] = None

        self._npc_vehicles: List[Any] = []

        self._npc_walkers: List[Any] = []

        self._sensors: List[Any] = []

        self._sensor_data = SensorBundle()



        self._step_count = 0

        self._episode_reward = 0.0

        self._total_dist_m = 0.0

        self._collision_count = 0

        self._lane_violations = 0

        self._prev_location = None

        self._episode_start = time.time()



        self._mock_speed = 0.0

        self._mock_steer = 0.0

        self._mock_lane_pos = 0.0



    def reset(

        self, *, seed: Optional[int] = None, options: Optional[dict] = None

    ) -> Tuple[dict, dict]:

        if seed is not None and GYM_AVAILABLE:

            super().reset(seed=seed)



        if CARLA_AVAILABLE:

            obs = self._carla_reset()

        else:

            obs = self._mock_reset()



        self._step_count = 0

        self._episode_reward = 0.0

        self._total_dist_m = 0.0

        self._collision_count = 0

        self._lane_violations = 0

        self._episode_start = time.time()



        return obs, {"scenario": self.scenario.name}



    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:

        steer = float(np.clip(action[0], -1.0, 1.0))

        throttle = float(np.clip(action[1], 0.0, 1.0))

        brake = float(np.clip(action[2], 0.0, 1.0))



        if CARLA_AVAILABLE and self._ego is not None:

            obs = self._carla_step(steer, throttle, brake)

        else:

            obs = self._mock_step(steer, throttle, brake)



        reward, terminated, info = self._compute_reward(obs)

        truncated = self._step_count >= self.MAX_EPISODE_STEPS



        self._step_count += 1

        self._episode_reward += reward

        return obs, reward, terminated, truncated, info



    def render(self) -> Optional[np.ndarray]:

        return self._sensor_data.rgb_image



    def close(self) -> None:

        self._cleanup()



    # ------------------------------------------------------------------

    # CARLA setup

    # ------------------------------------------------------------------



    def _carla_reset(self) -> dict:

        self._cleanup()

        cfg = SCENARIO_CONFIGS[self.scenario]



        try:

            self._client = carla.Client(self.host, self.port)

            self._client.set_timeout(self.timeout)



            # Reuse the running world. Do not call load_world().

            self._world = self._client.get_world()

        except Exception as exc:

            raise RuntimeError(

                f"[CarlaEnv] Could not connect to CARLA at "

                f"{self.host}:{self.port}. "

                f"Make sure CarlaUE4.exe is running and port 2000 is responding."

            ) from exc



        self._set_weather(cfg["weather"])



        # IMPORTANT: asynchronous mode is used on this laptop.

        settings = self._world.get_settings()

        settings.synchronous_mode = False

        settings.fixed_delta_seconds = None

        self._world.apply_settings(settings)



        # Spawn only the ego vehicle during startup.

        self._ego = self._spawn_ego()



        # IMPORTANT: Do not create the RGB camera while waiting for the OSM

        # destination. On this laptop, keeping PixelReader alive while the

        # browser/OSM step is idle can eventually trigger D3D device loss.

        # The camera is started explicitly after a destination is received.



        # Traffic Manager/NPCs are intentionally disabled.

        # They can be added back after the base driving loop is stable.

        # self._spawn_npcs(cfg["npcs"], cfg["peds"])



        self._prev_location = self._ego.get_location()

        return self._get_obs()



    def start_camera(self) -> None:

        """Start the low-resolution RGB camera after navigation begins."""

        if not self._sensors:

            self._attach_sensors()



    def _carla_step(self, steer: float, throttle: float, brake: float) -> dict:

        if self._ego is None:

            return self._mock_step(steer, throttle, brake)



        ctrl = carla.VehicleControl(

            throttle=throttle,

            steer=steer,

            brake=brake,

            hand_brake=False,

            reverse=False,

        )

        self._ego.apply_control(ctrl)



        # Never manually tick an asynchronous CARLA world.

        if self.synchronous:

            self._world.tick()



        self._update_distance()

        return self._get_obs()



    def _cleanup(self) -> None:

        # Stop camera listeners BEFORE destroying sensors.

        # This avoids leaving CARLA's PixelReader with an invalid render target.

        for sensor in list(self._sensors):

            try:

                sensor.stop()

            except Exception:

                pass



        for sensor in list(self._sensors):

            try:

                sensor.destroy()

            except Exception:

                pass



        for actor in list(self._npc_vehicles) + list(self._npc_walkers):

            try:

                actor.destroy()

            except Exception:

                pass



        if self._ego is not None:

            try:

                self._ego.destroy()

            except Exception:

                pass



        self._ego = None

        self._sensors = []

        self._npc_vehicles = []

        self._npc_walkers = []

        self._sensor_data = SensorBundle()



    def _spawn_ego(self):

        bp_lib = self._world.get_blueprint_library()

        bp = bp_lib.find("vehicle.tesla.model3")

        spawn_points = self._world.get_map().get_spawn_points()



        if not spawn_points:

            raise RuntimeError("CARLA has no spawn points.")
        candidates = list(spawn_points)
        random.shuffle(candidates)
        last_error = None
        for sp in candidates:
            try:
                ego = self._world.try_spawn_actor(bp, sp)
                if ego is not None:
                    ego.set_autopilot(False)
                    return ego
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not spawn ego vehicle. Last error: {last_error}")



    def _attach_sensors(self) -> None:

        """Attach the GPU-safe RGB camera.



        CARLA renders only 64x48. The callback upscales the image on CPU

        to 288x160 so the rest of the project keeps its existing API.

        """

        bp_lib = self._world.get_blueprint_library()



        cam_bp = bp_lib.find("sensor.camera.rgb")

        cam_bp.set_attribute("image_size_x", "64")

        cam_bp.set_attribute("image_size_y", "48")

        cam_bp.set_attribute("fov", "90")

        cam_bp.set_attribute("sensor_tick", "0.25")



        # Reduce GPU/rendering work where this CARLA build supports it.

        if cam_bp.has_attribute("enable_postprocess_effects"):

            cam_bp.set_attribute("enable_postprocess_effects", "False")



        cam_tf = carla.Transform(

            carla.Location(x=1.5, z=2.4)

        )



        cam = self._world.spawn_actor(

            cam_bp,

            cam_tf,

            attach_to=self._ego,

        )



        cam.listen(self._on_rgb)

        self._sensors.append(cam)



        print(f"[CarlaEnv] GPU-safe RGB camera attached: {cam.id}")



    def _spawn_npcs(self, n_vehicles: int, n_peds: int) -> None:

        # Disabled intentionally in the GPU-safe base configuration.

        self._npc_vehicles = []

        self._npc_walkers = []



    # ------------------------------------------------------------------

    # Sensor callbacks

    # ------------------------------------------------------------------



    def _on_rgb(self, image) -> None:

        # Keep this callback as light as the standalone camera_test.py.

        # Do NOT resize or run OpenCV here.

        arr = np.frombuffer(image.raw_data, dtype=np.uint8)

        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]

        self._sensor_data.rgb_image = arr[:, :, ::-1].copy()



    def _on_collision(self, event) -> None:

        self._sensor_data.collision_detected = True

        self._collision_count += 1



    def _on_lane_invasion(self, event) -> None:

        self._sensor_data.lane_invaded = True

        self._lane_violations += 1



    # ------------------------------------------------------------------

    # Observation

    # ------------------------------------------------------------------



    def _get_obs(self) -> dict:

        if CARLA_AVAILABLE and self._ego is not None:

            vel = self._ego.get_velocity()

            ctrl = self._ego.get_control()

            speed_kmh = 3.6 * math.sqrt(

                vel.x**2 + vel.y**2 + vel.z**2

            )

            steer = ctrl.steer

            lane_pos = self._get_lane_offset()

        else:

            speed_kmh = self._mock_speed

            steer = self._mock_steer

            lane_pos = self._mock_lane_pos



        img = self._sensor_data.rgb_image

        if img is None:

            img = np.zeros((48, 64, 3), dtype=np.uint8)

        elif img.shape[:2] != (self.IMG_H, self.IMG_W):

            # Resize outside the CARLA sensor callback, on the Python side.

            try:

                import cv2

                img = cv2.resize(

                    img,

                    (self.IMG_W, self.IMG_H),

                    interpolation=cv2.INTER_LINEAR,

                )

            except Exception:

                y_idx = np.linspace(

                    0, img.shape[0] - 1, self.IMG_H

                ).astype(np.int32)

                x_idx = np.linspace(

                    0, img.shape[1] - 1, self.IMG_W

                ).astype(np.int32)

                img = img[y_idx][:, x_idx]



        return {

            "camera": img.astype(np.uint8),

            "speed": np.array([speed_kmh], dtype=np.float32),

            "steer": np.array([steer], dtype=np.float32),

            "lane_pos": np.array([lane_pos], dtype=np.float32),

        }



    def _get_lane_offset(self) -> float:

        if self._ego is None or self._world is None:

            return 0.0



        try:

            wp = self._world.get_map().get_waypoint(

                self._ego.get_location(),

                project_to_road=True,

            )

            if wp is None:

                return 0.0



            loc = self._ego.get_location()

            wp_loc = wp.transform.location

            dx = loc.x - wp_loc.x

            dy = loc.y - wp_loc.y

            return float(math.hypot(dx, dy))

        except Exception:

            return 0.0



    def _update_distance(self) -> None:

        if self._ego and self._prev_location:

            loc = self._ego.get_location()

            self._total_dist_m += loc.distance(self._prev_location)

            self._prev_location = loc



    # ------------------------------------------------------------------

    # Reward

    # ------------------------------------------------------------------



    def _compute_reward(self, obs: dict) -> Tuple[float, bool, dict]:

        reward = 0.0

        terminated = False



        speed = float(obs["speed"][0])

        lane_pos = abs(float(obs["lane_pos"][0]))

        collided = self._sensor_data.collision_detected

        lane_inv = self._sensor_data.lane_invaded



        self._sensor_data.collision_detected = False

        self._sensor_data.lane_invaded = False



        if lane_pos < 0.5:

            reward += 1.0

        else:

            reward -= 0.2 * lane_pos



        reward += 0.01 * min(speed, 50.0)



        if collided:

            reward -= 5.0

            terminated = True



        if lane_inv:

            reward -= 3.0



        info = {

            "speed": speed,

            "lane_pos": lane_pos,

            "collision": collided,

            "total_dist_m": self._total_dist_m,

            "episode_reward": self._episode_reward,

            "step": self._step_count,

        }

        return reward, terminated, info



    # ------------------------------------------------------------------

    # Weather

    # ------------------------------------------------------------------



    def _set_weather(self, weather_preset: str) -> None:

        if self._world is None:

            return



        presets = {

            "ClearNoon": carla.WeatherParameters.ClearNoon,

            "CloudyNoon": carla.WeatherParameters.CloudyNoon,

            "ClearNight": carla.WeatherParameters.ClearNight,

        }



        if weather_preset in presets:

            self._world.set_weather(presets[weather_preset])



    # ------------------------------------------------------------------

    # Mock mode

    # ------------------------------------------------------------------



    def _mock_reset(self) -> dict:

        self._mock_speed = 0.0

        self._mock_steer = 0.0

        self._mock_lane_pos = 0.0

        self._sensor_data = SensorBundle()



        return {

            "camera": np.zeros(

                (self.IMG_H, self.IMG_W, 3),

                dtype=np.uint8,

            ),

            "speed": np.array([0.0], dtype=np.float32),

            "steer": np.array([0.0], dtype=np.float32),

            "lane_pos": np.array([0.0], dtype=np.float32),

        }



    def _mock_step(self, steer, throttle, brake) -> dict:

        acc = (throttle - brake) * 5.0

        self._mock_speed = max(

            0.0,

            min(

                120.0,

                self._mock_speed + acc * self.delta_t,

            ),

        )

        self._mock_steer = steer

        self._mock_lane_pos = abs(steer) * 0.5



        return {

            "camera": np.random.randint(

                0, 255,

                (self.IMG_H, self.IMG_W, 3),

                dtype=np.uint8,

            ),

            "speed": np.array(

                [self._mock_speed],

                dtype=np.float32,

            ),

            "steer": np.array(

                [steer],

                dtype=np.float32,

            ),

            "lane_pos": np.array(

                [self._mock_lane_pos],

                dtype=np.float32,

            ),

        }



    @property

    def metrics(self) -> dict:

        return {

            "step": self._step_count,

            "episode_reward": self._episode_reward,

            "total_dist_m": self._total_dist_m,

            "collisions": self._collision_count,

            "lane_violations": self._lane_violations,

            "elapsed_s": time.time() - self._episode_start,

        }





if __name__ == "__main__":

    env = CarlaEnv(scenario=Scenario.STRAIGHT_ROAD)

    obs, info = env.reset()



    print("=" * 50)

    print("CARLA Environment Test")

    print(f"  Obs keys : {list(obs.keys())}")

    print(f"  Camera   : {obs['camera'].shape}")

    print(f"  Speed    : {obs['speed']}")



    for i in range(10):

        action = np.array([0.0, 0.5, 0.0])

        obs, reward, term, trunc, info = env.step(action)

        print(

            f"  Step {i+1:02d} | "

            f"speed={obs['speed'][0]:.1f} kmh | "

            f"reward={reward:.3f}"

        )

        if term:

            break



    print(f"\nMetrics: {env.metrics}")

    env.close()

    print("=" * 50)
