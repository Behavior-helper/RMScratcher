# !/bin/bash 
timestamp=`date +"%Y%m%d%H%M"`
gst-launch-1.0 -v nvv4l2camerasrc device=/dev/video$1 \
! "video/x-raw(memory:NVMM),format=(string)UYVY, width=(int)640, height=(int)480,framerate=(fraction)120/1" \
! nvvidconv \
! "video/x-raw(memory:NVMM),format=(string)NV12" \
! nvv4l2h264enc control-rate=1 bitrate=8000000 \
! h264parse \
! splitmuxsink location=$2/$timestamp-$1_%02d.mkv muxer-factory=matroskamux max-size-time=300000000000 muxer-properties="properties,async-handling=1" \
-e
