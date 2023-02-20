
import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from nets.ournet import *
from PIL import Image
import scipy.misc


class PointTracker(object):
  def __init__(self, nn_thresh, desc1, pts1):
    self.nn_thresh = nn_thresh
    self.all_pts = []
    # for n in range(self.maxl):
    #   self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.desc1 = desc1
    self.pts1 = pts1
    self.matches = []
    self.all_pts.append(self.pts1)
    self.all_pts.append(self.pts1)

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    #print(desc1.shape)
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)   # N * M
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)  # 找到每列的最小值的位置
    #print(idx.shape)
    scores = dmat[np.arange(dmat.shape[0]), idx]  # 对应位置的得分图
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1  # 位置索引
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.
    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """

    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    self.all_pts.pop(1)
    self.all_pts.append(pts)
    matches = self.nn_match_two_way(self.desc1, desc, self.nn_thresh)
    self.matches = matches.T
    return

  def sigle_rotate_compute(self, pre_point, last_point, pre_center, last_center, rotate_truth):  # 计算两张图之间的旋转角度
    pre_arr = np.array(pre_point) - np.array(pre_center)
    last_arr = np.array(last_point) - np.array(last_center)
    v1 = pre_arr
    v2 = last_arr
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
    cos_value = np.dot(v1, v2) / TheNorm
    cos_value = np.clip(cos_value, -1, 1)
    theta = np.rad2deg(np.arccos(cos_value))
    if rho < 0:
      theta = -theta
    if abs(theta - rotate_truth) < 2:
      return (0, 255, 0)
    else:
      return (255, 0, 0)

  ##计算向量、旋转部分
  def array_rotate_compute(self, pre_points, last_points, pre_center, last_center, filter_thresh):  # 计算两张图之间的旋转角度
    if not pre_points or not last_points:
      return 0
    pre_arr = np.array(pre_points) - np.array(pre_center)
    last_arr = np.array(last_points) - np.array(last_center)
    ls = []
    for i in range(len(pre_arr)):
      v1 = pre_arr[i]
      v2 = last_arr[i]
      TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
      rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
      cos_value = np.dot(v1, v2) / TheNorm
      cos_value = np.clip(cos_value, -1, 1)
      theta = np.rad2deg(np.arccos(cos_value))
      if rho < 0:
        theta = -theta
      ls.append(theta)
    ls = np.array(ls)
    retain = (abs(np.array(ls) - np.median(ls)) < filter_thresh)  # T=20
    ls = np.array(ls)[retain]
    if np.sum(retain) == 0:
      return 0
    else:
      return np.median(ls)

  def draw_tracks(self, tracks, oval0, oval1, filter_thresh):

    # Store the number of points per camera.
    pts_mem = self.all_pts

    # Iterate through each track and draw it.
    self.point_list_pre = []
    self.point_list_last = []
    for track in tracks:
      idx0 = int(track[0])
      idx1 = int(track[1])
      pt0 = pts_mem[0][:2, idx0]
      pt1 = pts_mem[1][:2, idx1]
      p0 = (int(round(pt0[0])), int(round(pt0[1])))
      p1 = (int(round(pt1[0])), int(round(pt1[1])))
      self.point_list_pre.append(p0)
      self.point_list_last.append(p1)
    rotate = self.array_rotate_compute(self.point_list_pre, self.point_list_last, pre_center=(oval0[0], oval0[1]), last_center=(oval1[0], oval1[1]), filter_thresh=filter_thresh)
    print('Rotate:{:.5f}'.format(rotate))
    return rotate

class VideoStreamer(object):
  def __init__(self, basedir, camid, height, width, skip, img_glob, oval_path):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    self.oval_dict = {}
    self.genOvaldict(oval_path)
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        print(img_glob)
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort(key=lambda x : int((x.split('/')[-1]).split('.')[0]))
        #print(self.listing)
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')


  def read_image(self, impath, img_size):
    img = Image.open(impath)
    img= img.resize((img_size[1],img_size[0])).convert('RGB')
    return img


  def genOvaldict(self, file_path):
    file = open(file_path)
    lines = file.readlines()
    for line in lines:
      self.oval_dict[line.split(';')[0]] = [float(line.split(';')[1]), float(line.split(';')[2]),\
                                            float(line.split(';')[3]), float(line.split(';')[4].strip('/n'))]
    return self.oval_dict

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    oval_index = None
    image_id = None
    if self.i == self.maxlen:
      return (None, False, None, None)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False, None, None)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
      image_id = image_file.split('/')[-1]
      oval_index = self.oval_dict[image_id]

    # Increment internal counter.
    self.i = self.i + 1
    #input_image = input_image.astype('float32')
    return (input_image, True, oval_index, image_id)


def getRotateDict(txt_path):
  file = open(txt_path, 'r')
  lines = file.readlines()
  rotate_dict = {}
  for line in lines:
    rotate_dict[line.split(':')[0]] = line.split(':')[1].strip('\n')
  return rotate_dict


def findLimbusPoint(kpts, desc, oval_dict, gain_w):
  kpts = kpts.transpose()
  desc = desc.transpose()
  xc = oval_dict[0]
  yc = oval_dict[1]
  Rmin = min(oval_dict[2] / 2, oval_dict[3] / 2)
  Rmax = max(oval_dict[2] / 2, oval_dict[3] / 2) + gain_w
  keep = []

  for i in range(len(kpts[1])):
    dis = pow(kpts[0][i] - xc, 2) + pow(kpts[1][i] - yc, 2)
    if dis > pow(Rmin, 2) and dis < pow(Rmax, 2):
      keep.append(True)
    else:
      keep.append(False)
  return kpts[:, keep], desc[:, keep]


def load_network(model_fn):
  checkpoint = torch.load(model_fn)
  print("\n>> Creating net = " + checkpoint['net'])
  net = eval(checkpoint['net'])
  nb_of_weights = common.model_size(net)
  print(f" ( Model size: {nb_of_weights / 1000:.0f}K parameters )")

  # initialization
  weights = checkpoint['state_dict']
  net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
  return net.eval()


class NonMaxSuppression(torch.nn.Module):
  def __init__(self, rel_thr=0.7, rep_thr=0.3):
    nn.Module.__init__(self)
    self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.rel_thr = rel_thr
    self.rep_thr = rep_thr

  def forward(self, reliability, repeatability, **kw):
    assert len(reliability) == len(repeatability) == 1
    reliability, repeatability = reliability[0], repeatability[0]

    # local maxima
    maxima = (repeatability == self.max_filter(repeatability))

    # remove low peaks
    maxima *= (repeatability >= self.rep_thr)
    maxima *= (reliability >= self.rel_thr)

    return maxima.nonzero().t()[2:4]


def run_net(net, img, detector):
  with torch.no_grad():
    res = net(imgs=[img])
  # get output and reliability map
  descriptors = res['descriptors'][0]
  y, x = detector(**res)  # nms
  d = descriptors[0, :, y, x].t()
  xy = torch.stack([x, y], dim=-1)
  return xy, d

if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('--input', type=str, default='data/Cataract_Test/Video/', help='Image directory')
  parser.add_argument('--oval_path', type=str, default='data/Cataract_Test/Oval_index/', help='oval index txt abspath')
  parser.add_argument('--write_dir', type=str, default='output/predict/Video/', help='Directory where to write output frames.')
  parser.add_argument('--txt_result', type=str, default='output/predict/Txt/', help='Directory where to write output txt.')
  parser.add_argument('--eval_result', type=str, default='output/predict/FPS.txt', help='Directory where to write output txt.')
  #parser.add_argument('--weights_path', type=str, default='checkpoints/ours.pt', help='ours checkpoints')
  parser.add_argument('--weights_path', type=str, default='checkpoints/r2d2.pt', help='r2d2 checkpoints')

  parser.add_argument('--H', type=int, default=540, help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=960, help='Input image width (default:160).')

  parser.add_argument('--gain_w', type=int, default=30, help='W')
  parser.add_argument('--filter_thresh', type=float, default=20, help='T')
  parser.add_argument('--nn_thresh', type=float, default=0.85, help='nn_thresh')

  parser.add_argument('--cuda', type=bool, default=True, help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--camid', type=int, default=0, help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1, help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--get_time', type=bool, default=True, help='compute time cost')
  parser.add_argument('--img_glob', type=str, default='*.jpg', help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1, help='Images to skip if input is movie or directory (default: 1).')

  opt = parser.parse_args()
  print(opt)

  total_net_FPS = 0
  total_total_FPS = 0
  time_i = 0

  vid_list = sorted(os.listdir(opt.input))
  device = torch.device("cuda:0" if opt.cuda else "cpu")
  if not os.path.exists(opt.txt_result):
    os.makedirs(opt.txt_result)
  for vid in vid_list:
    seq_list = sorted(os.listdir(os.path.join(opt.input, vid)))
    for seq in seq_list:
      seq_path = os.path.join(opt.input, vid, seq)
      oval_path = os.path.join(opt.oval_path, seq+'.txt')
      save_image_path = os.path.join(opt.write_dir, vid, seq)
      save_txt_path = os.path.join(opt.txt_result, seq+'.txt')

      # This class helps load input images from different sources.
      vs = VideoStreamer(seq_path, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob, oval_path)
      
      print('==> Loading network.')
      net = load_network(opt.weights_path)
      if opt.cuda:
        net = net.cuda()
      # create the non-maxima detector
      detector = NonMaxSuppression()

      print('==> Running Demo.')
      # 创建t0xt结果文件
      file = open(save_txt_path, 'w')

      # This class helps merge consecutive point matches into tracks.
      img0, status, oval_index0, image_id = vs.next_frame()

      # adjust the region of img
      scale = 1920/opt.W
      oval0 = [i/scale for i in oval_index0]
      R = max(oval0[2], oval0[3]) / 2
      bandwidth = opt.gain_w
      x0 = int(max(oval0[0] - R - bandwidth, 0))
      x1 = int(min(oval0[0] + R + bandwidth, opt.W))
      y0 = int(max(oval0[1] - R - bandwidth, 0))
      y1 = int(min(oval0[1] + R + bandwidth, opt.H))
      img0 = img0.crop((x0,y0,x1,y1))
      image0 = img0
      image0 = cv2.cvtColor(np.asarray(image0), cv2.COLOR_RGB2BGR)
      img0  = norm_RGB(img0)[None]

      print("frame_size:{}".format(img0.shape))
      oval0[0] = min(oval0[0], R + bandwidth)
      oval0[1] = min(oval0[1], R + bandwidth)
      if opt.cuda:
        img0 = img0.cuda()
      pts0, desc0 = run_net(net, img0, detector)
      pts0, desc0 = findLimbusPoint(pts0.cpu().numpy(), desc0.cpu().numpy(), oval0, opt.gain_w)
      tracker = PointTracker(nn_thresh=opt.nn_thresh, desc1=desc0, pts1=pts0)

      while True:
        # Get a new image.

        img, status, oval_index, image_id = vs.next_frame()
        if status is False:
          break
        print(vid + '_' + seq + '_' + image_id)

        start = time.time()
        ## 眼球区域裁剪
        oval1 = [i / scale for i in oval_index]
        x0 = int(max(oval1[0] - R - bandwidth, 0))
        x1 = int(min(oval1[0] + R + bandwidth, opt.W))
        y0 = int(max(oval1[1] - R - bandwidth, 0))
        y1 = int(min(oval1[1] + R + bandwidth, opt.H))
        img = img.crop((x0, y0, x1, y1))
        img = norm_RGB(img)[None]
        print("img_size:{}".format(img.shape))
        oval1[0] = min(oval1[0], R + bandwidth)
        oval1[1] = min(oval1[1], R + bandwidth)

        ## 特征点提取与描述（net）
        start_net = time.time()
        if opt.cuda:
          img = img.cuda()
        pts1, desc1 = run_net(net, img, detector)
        end_net = time.time()

        ## 目标匹配区域确定及特征点筛选
        pts1, desc1 = findLimbusPoint(pts1.cpu().numpy(), desc1.cpu().numpy(), oval1, opt.gain_w)

        ## 特征点匹配
        tracker.update(pts1, desc1)
        tracks = tracker.matches

        ## 眼球旋转角度计算(包括异常匹配点剔除)
        rotate = tracker.draw_tracks(tracks=tracks, oval0=oval0, oval1=oval1, filter_thresh=opt.filter_thresh)
        file.writelines(image_id+':'+str(rotate)+'\n')
        end = time.time()

        ## Timer >> FPS
        net_t = (1./ float(end_net-start_net))
        total_t = (1./ float(end-start))
        total_net_FPS += net_t
        total_total_FPS += total_t
        time_i += 1
        if opt.get_time:
          print('Processed image %d (net: %.2f FPS, total: %.2f FPS).' % (vs.i, net_t, total_t))
          print('-'*50)
      file.close()
      print('==> Finshed Test.')

  Mean_net_FPS = total_net_FPS / time_i
  Mean_total_FPS = total_total_FPS / time_i
  print('Mean_net_FPS:{}\nMean_total_FPS:{}'.format(Mean_net_FPS, Mean_total_FPS))
  FPS_file = open(os.path.join(opt.eval_result), 'w+')
  FPS_file.writelines('Mean_net_FPS:' + str(Mean_net_FPS) + '\n')
  FPS_file.writelines('Mean_total_FPS:' + str(Mean_total_FPS) + '\n')
  FPS_file.close()
