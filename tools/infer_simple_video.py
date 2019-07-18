#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import subprocess as sp

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--video',
        dest='video_fn',
        help='video file name.',
        type=str,
	required = True
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--output',
        dest='output_fn',
        help='output video file name (default: output.mp4)',
        default='/tmp/infer_simple_vid',
        type=str
    )
    parser.add_argument(
	'--downsize',
	dest='downsize',
	help='divide ratio of video. downsized before network forward.',
	type=int,
	default=1
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    vid_cap = cv2.VideoCapture(args.video_fn)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)/args.downsize)
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/args.downsize)

    #fourcc = cv2.VideoWriter_fourcc(*'XIVD')
    #vid_out = cv2.VideoWriter('output.avi',-1, 20.0, (width,height))
    
    FFMPEG_BIN = "ffmpeg"
    command = [FFMPEG_BIN,
	'-y', # overwrite output
	'-f', 'rawvideo',
	'-vcodec', 'rawvideo',
	'-s', '{}x{}'.format(width, height),
	'-pix_fmt', 'rgb24',
	'-r', '25', # fps
	'-i', '-', # input comes from a pipe
	'-an', # no audio
	'-vcodec', 'mpeg4',
	'-b:v', '30M',
	'-maxrate', '30M',
	'-loglevel', 'panic',
	'-nostdin',
	args.output_fn
	]
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    print(command)
    ret = True
    fid = 0
    while ret:
	print("time:",vid_cap.get(cv2.CAP_PROP_POS_MSEC)/1000,"sec")
	ret, im = vid_cap.read()
	if not ret:
	    break
	im = cv2.resize(im, dsize = (width,height), interpolation = cv2.INTER_LINEAR)
	with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, None
            )
	rst_im = vis_utils.vis_one_image_opencv(
	    im,
	    cls_boxes,
	    segms = cls_segms,
	    keypoints = cls_keyps,
	    thresh = args.thresh,
	    kp_thresh = args.kp_thresh,
	    show_box = True,
	    dataset = dummy_coco_dataset,
	    show_class = True
	)
	rgb_im = cv2.cvtColor(rst_im, cv2.COLOR_BGR2RGB)
	cv2.imshow('frame', rst_im)
	pipe.stdin.write(rgb_im.tostring())
	#pipe.stdin.flush()
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

    vid_cap.release()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
