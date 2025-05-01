import sys
from pathlib import Path
import cv2
import argparse
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (check_img_size, increment_path, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

def run(source='0', weights='yolov5s.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, max_det=1000, device='', output='output'):
    print("Starting script...")

    # Initialize
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img_size, s=stride)

    # Dataloader
    if source == '0':  # Check if source is webcam
        source = int(source)  # Convert '0' string to int for webcam ID
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Define output video writer
    output_path = f'{output}/output_video.avi'
    print(f"Output video will be saved to: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None  # Initialize 'out' variable later after getting the frame dimensions

    # Run inference
    model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    frame_count = 0
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0, 43], agnostic=False, max_det=max_det)

        # Process detections
        for i, det in enumerate(pred):
            if source == '0':  # If webcam, handle multiple streams
                p, im0 = path[i], im0s[i].copy()
                s += f'{i}: '
            else:
                p, im0 = path, im0s.copy()

            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            # Stream results
            im0 = annotator.result()
            if source == '0':  # If webcam, show each stream
                dataset.streams[i].imshow(im0)

            # Initialize video writer with correct frame size
            if out is None:
                height, width, _ = im0.shape
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                if not out.isOpened():
                    print("Failed to open video writer")
                    return

            # Write annotated frame to output video
            out.write(im0)
            frame_count += 1
            print(f"Writing frame {frame_count} to {output_path}")

    # Release video writer
    out.release()
    print(f"Finished writing video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', type=str, default='output', help='output directory')
    opt = parser.parse_args()
    print(opt)

    run(opt.source, opt.weights, opt.img_size, opt.conf_thres, opt.iou_thres, opt.max_det, opt.device, opt.output)
