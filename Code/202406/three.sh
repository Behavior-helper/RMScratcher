FWIDTH=`xrandr | grep '*' | tr -s [:space:] | awk -F '[\tx ]'  '{print $2}'`
((WIDTH=FWIDTH/2))
WIDTH=640

FHEIGHT=`xrandr | grep '*' | tr -s [:space:] | awk -F '[\tx ]'  '{print $3}'` 
((HEIGHT=FHEIGHT/2))
XWIDTH=`echo "$FHEIGHT / 0.75" | bc`
HEIGHT=480


CAPS="video/x-raw(memory:NVMM), width=$WIDTH, height=$HEIGHT, format=(string)UYVY, framerate=(fraction)120/1"

gst-launch-1.0 nvcompositor name=comp  \
sink_0::xpos=0 sink_0::ypos=0 sink_0::width=$WIDTH sink_0::height=$HEIGHT  \
sink_1::xpos=0 sink_1::ypos=$HEIGHT sink_1::width=$WIDTH sink_1::height=$HEIGHT \
sink_2::xpos=$WIDTH sink_2::ypos=0 sink_2::width=$WIDTH sink_2::height=$HEIGHT \
sink_3::xpos=$WIDTH  sink_3::ypos=$HEIGHT sink_3::width=$WIDTH sink_3::height=$HEIGHT \
! nvoverlaysink  overlay-w=$XWIDTH overlay-h=$FHEIGHT overlay-x=500   \
nvv4l2camerasrc device=/dev/video0 ! $CAPS ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)RGBA" \
! comp.  \
nvv4l2camerasrc device=/dev/video1 ! $CAPS ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)RGBA" \
! comp.  \
nvv4l2camerasrc device=/dev/video2 ! $CAPS ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)RGBA" \
! comp. \
nvv4l2camerasrc device=/dev/video3 ! $CAPS ! nvvidconv ! "video/x-raw(memory:NVMM),format=(string)RGBA" \
! comp. 