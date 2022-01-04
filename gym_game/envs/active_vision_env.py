from enum import Enum

import numpy as np
from gym import spaces
from skimage import img_as_float32

from utils.image_utils import fast_resize, to_pytorch_from_uint8
from utils.writer_singleton import WriterSingleton
from .pygame_env import PyGameEnv

import logging
logger = logging.getLogger("ActiveVisionEnv")
logger.setLevel("INFO")

"""
  ActiveVisionEnv receives a screen image.
  All other quantities are expressed relative to that screen image. 
"""


class GazeMode(Enum):
  ABSOLUTE = 1
  ITERATIVE = 2


class GridUtil:
  """
    Row major
    (0,0) is top left
  """

  def __init__(self, grid_length_x, grid_length_y, screen_height, screen_width, offset):
    self.grid_length_x = grid_length_x
    self.grid_length_y = grid_length_y
    self.screen_height = screen_height
    self.screen_width = screen_width

    self.grid_length = np.array([grid_length_x, grid_length_y])
    self.screen_size = np.array([screen_width, screen_height])
    self.grid_cell_size = np.array([screen_width / grid_length_x, screen_height / grid_length_y])

    self.offset = offset

  def action_2_xy(self, action):
    """
      input action: integer represented an absolute position
      return: type ndarray, (x,y) coordinate in pixel space
    """

    action = action - self.offset

    xy_grid = np.array([action % self.grid_length_x, action // self.grid_length_x])   # x, y
    xy_coord = np.multiply(xy_grid, self.grid_cell_size) + 0.5 * self.grid_cell_size
    return xy_coord.astype(int)

  def xy_2_action(self, xy_coord):
    """
      input xy_coord: type ndarray, (x,y) coordinate in pixel space
      return: action, integer represented an absolute position
    """

    xy_grid = np.floor(np.divide(xy_coord, self.grid_cell_size))
    x = xy_grid[0]
    y = xy_grid[1]
    action = y * self.grid_length_x + x
    action = int(action + self.offset)
    return action

  def num_cells(self):
    return self.grid_length_x * self.grid_length_y


class ActiveVisionEnv(PyGameEnv):
  metadata = {'render.modes': ['human', 'rgb_array']}

  HUMAN = 'human'
  ARRAY = 'rgb_array'
  GAZE_CONTROL_MODE = GazeMode.ABSOLUTE
  STEP = 0

  def __init__(self, num_actions, screen_width, screen_height, frame_rate):
    """
    The concrete environment (e.g. dm2s-v0) is responsible for reading in the config in its own format
    and creating a dictionary of params before calling super.init() to get to here.
    i.e. the params are available via self.get_config()
    """
    logger.info("IN")
    config = self.get_config()

    self.screen_scale = float(config["screen_scale"])  # resize the screen image before returning as an observation
    self.summaries = config['summaries']

    self.enabled = False if config["enable_active_vision"] == 0 else True
    if not self.enabled:
      super().__init__(num_actions, screen_width, screen_height, frame_rate)
      return

    self.fov_fraction = float(config["fovea_fraction"])  # fovea occupies this fraction of the screen image (applied to x and y respectively)
    self.fov_scale = float(config["fovea_scale"])  # image size, expressed as fraction of screen size
    self.step_size = int(config["gaze_step_size"])  # step size of gaze movement, in pixels in the screen image
    self.peripheral_scale = float(config["peripheral_scale"])  # peripheral image size, expressed as fraction of screen size
    self.peripheral_noise_magnitude = float(config["peripheral_noise_magnitude"])

    self.fov_size = np.array([int(self.fov_fraction * screen_width), int(self.fov_fraction * screen_height)])
    self.fov_size_half = 0.5*self.fov_size
    self.gaze_centre = np.array([screen_width // 2, screen_height // 2])  # gaze position - (x, y)--> *center of fovea*

    self._show_fovea = True if config["show_fovea_on_screen"] else False

    self.grid_utils = None
    self.grid_length_x = None
    self.grid_length_y = None
    self._action_2_xy = None

    if self.GAZE_CONTROL_MODE is GazeMode.ITERATIVE:
      self._action_2_xy = {  # map actions (integers) to x,y gaze delta
        num_actions: np.array([-1, 0]),      # 3. left
        num_actions + 1: np.array([1, 0]),   # 4. right
        num_actions + 2: np.array([0, -1]),  # 5. up
        num_actions + 3: np.array([0, 1])    # 6. down
      }
      self._actions_start = num_actions
      self._actions_end = num_actions + len(self._action_2_xy)
    else:
      self.grid_length_x = config["grid_length_x"]
      self.grid_length_y = config["grid_length_y"]
      self.grid_utils = GridUtil(self.grid_length_x, self.grid_length_y, screen_height, screen_width,
                                 offset=num_actions)

      self._actions_start = num_actions
      self._actions_end = num_actions + self.grid_utils.num_cells()

    self._img_fov = None
    self._img_periph = None
    self._img_full = None
    self._img_full_nw = None

    # bounds for centre of gaze
    self._x_min = self.fov_size_half[0]
    self._x_max = screen_width - self.fov_size_half[0]
    self._y_min = self.fov_size_half[1]
    self._y_max = screen_height - self.fov_size_half[1]
    # self.i = 0      # used for debugging

    super().__init__(num_actions, screen_width, screen_height, frame_rate)

  def reset(self):
    """Reset gaze coordinates"""
    return super().reset()

  def _do_step(self, action, time):
    # update the position of the fovea (fov_pos), given the action taken
    logger.debug("Received action: %s", action)

    if not self.enabled:
      return

    # if within action scope, modify gaze
    if self._actions_start <= action < self._actions_end:

      if self.GAZE_CONTROL_MODE is GazeMode.ITERATIVE:
        self.gaze_centre = self.gaze_centre + self._action_2_xy[action] * self.step_size
      else:
        self.gaze_centre = self.grid_utils.action_2_xy(action)

      self.gaze_centre[0] = np.clip(self.gaze_centre[0], self._x_min, self._x_max)   # ensure x coord is in bounds
      self.gaze_centre[1] = np.clip(self.gaze_centre[1], self._y_min, self._y_max)   # ensure y coord is in bounds

      #print("New gaze: ", self.gaze)

  def _create_action_space(self, num_actions):
    if self.enabled:
      total_actions = self._actions_end  # Gaze control
    else:
      total_actions = num_actions  # in this case, do not add any actions

    self.action_space = spaces.Discrete(total_actions)

  def _create_observation_space(self, screen_width, screen_height, channels=3):
    if not self.enabled:
      full_shape = self.get_full_observation_shape()
      full = spaces.Box(low=0, high=255, shape=full_shape, dtype=np.uint8)
      self.observation_space = spaces.Dict({'full': full})
    else:
      fovea_shape = self.get_fovea_observation_shape()  #(channels, screen_height, screen_width)
      peripheral_shape = self.get_peripheral_observation_shape()  #(channels, screen_height, screen_width)
      fovea = spaces.Box(low=0, high=1.0, shape=fovea_shape, dtype=np.float32)
      peripheral = spaces.Box(low=0, high=1.0, shape=peripheral_shape, dtype=np.float32)
      gaze = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([screen_width, screen_height]), dtype=np.float32)
      self.observation_space = spaces.Dict({
        'fovea': fovea,
        'peripheral': peripheral,
        'gaze': gaze})

  def get_full_observation_shape(self):
    h = self.screen_shape[0]
    w = self.screen_shape[1]
    c = self.screen_shape[2]
    h2 = int(h * self.screen_scale)
    w2 = int(w * self.screen_scale)
    return c, h2, w2

  def get_fovea_observation_shape(self):
    h = self.screen_shape[0]
    w = self.screen_shape[1]
    c = self.screen_shape[2]
    pixels_h = int(h * self.fov_fraction)
    pixels_w = int(w * self.fov_fraction)
    h2 = int(pixels_h * self.fov_scale)
    w2 = int(pixels_w * self.fov_scale)
    return c, h2, w2

  def get_peripheral_observation_shape(self):
    h = self.screen_shape[0]
    w = self.screen_shape[1]
    c = self.screen_shape[2]
    h2 = int(h * self.peripheral_scale)
    w2 = int(w * self.peripheral_scale)
    return c, h2, w2

  def get_observation(self):
    """
    The observation is a render of the screen.
    The format is ndarray of 8 bit unsigned integers
    Return images as ndarray, in PyTorch format: 32 bit floating point images
    """

    self.STEP += 1

    debug = False

    img = self.render(mode='rgb_array')

    #from timeit import default_timer as timer
    #start = timer()

    img = np.transpose(img, [1, 0, 2])  # observed img is horizontally reflected, and rotated 90 deg ...
    img_shape = img.shape

    if not self.enabled:

      # resize screen image before returning as observation
      img_resized = fast_resize(img, self.screen_scale)

      # convert to PyTorch format
      self._img_full_nw = to_pytorch_from_uint8(img_resized)

      writer = WriterSingleton.get_writer()
      if self.summaries and writer:
        import torch

        input_img_pt = to_pytorch_from_uint8(img)

        img_tensor = torch.tensor(input_img_pt)
        img_full_tensor = torch.tensor(self._img_full_nw)

        writer.add_image('active-vision-nw/input', img_tensor, global_step=self.STEP)
        writer.add_image('active-vision-nw/full', img_full_tensor, global_step=self.STEP)
        writer.add_histogram('active-vision-nw/hist-full', img_full_tensor, global_step=self.STEP)
        writer.flush()

      # Assemble dict
      observation = {
        'full': self._img_full_nw,
      }

    else:

      # Peripheral Image - downsize to get peripheral (lower resolution) image
      img_periph = fast_resize(img, self.peripheral_scale)

      # Foveal Image - crop to fovea and rescale
      h, w, ch = img.shape[0], img.shape[1], img.shape[2]
      pxl_h_half = int(0.5 * h * self.fov_fraction)
      pxl_w_half = int(0.5 * w * self.fov_fraction)

      self._img_fov = img[self.gaze_centre[1] - pxl_h_half:self.gaze_centre[1] + pxl_h_half,
                          self.gaze_centre[0] - pxl_w_half:self.gaze_centre[0] + pxl_w_half]
      self._img_fov = fast_resize(self._img_fov, self.fov_scale)

      # convert to pytorch format
      self._img_fov = to_pytorch_from_uint8(self._img_fov)
      img_periph = to_pytorch_from_uint8(img_periph)

      # add noise to peripheral image
      img_periph_random = (np.random.random(img_periph.shape)-0.5)*self.peripheral_noise_magnitude
      self._img_periph = np.clip(img_periph + img_periph_random, a_min=0.0, a_max=1.0).astype(np.float32)

      # print('fovea shape trans:', self._img_fov.shape)

      # debugging
      if debug:
        print('img orig screen shape:', img_shape)
        print('img periph shape:', self._img_periph.shape)
        print('img fovea shape:', self._img_fov.shape)
        print('img full (rescaled) shape:', self._img_full.shape)

      writer = WriterSingleton.get_writer()
      if self.summaries and writer:
        import torch
        img = to_pytorch_from_uint8(img)
        writer.add_image('active-vision/input', torch.tensor(img), global_step=self.STEP)
        writer.add_image('active-vision/fovea', torch.tensor(self._img_fov), global_step=self.STEP)
        writer.add_image('active-vision/peripheral', torch.tensor(self._img_periph), global_step=self.STEP)
        writer.flush()

      # Assemble dict
      observation = {
        'full': self._img_full,
        'fovea': self._img_fov,
        'peripheral': self._img_periph,
        'gaze': self.gaze_centre.astype(np.float32)
      }

    #end = timer()
    #print('Step obs: ', str(end - start)) # Time in seconds, e.g. 5.38091952400282

    return observation

  def draw_screen(self, screen, screen_options):
    import pygame as pygame
    # draw the gaze position
    if self.enabled and self._show_fovea:
      BLACK = (0, 0, 0)
      # gaze pos is top left of fovea
      fovea_rect = pygame.Rect(self.gaze_centre[0] - self.fov_size_half[0], self.gaze_centre[1] - self.fov_size_half[1],
                               self.fov_size[0], self.fov_size[1])
      pygame.draw.rect(screen, BLACK, fovea_rect, 1)
