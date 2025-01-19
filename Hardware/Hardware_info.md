# Hardware Solutions

## CS-TX2-XAVIER-nCAM module
### Overview
CS-TX2-XAVIER-nCAM module solution. This is a multi-channel MIPI CSI camera module specifically designed for compatibility with Jetson TX2. The module consists of three parts: 
1) ADP-N1-V2.0, a four-channel MIPI CSI interface adapter board that connects to the MIPI CSI interface of the Jetson series; 
2) ADP-U1, which converts the FPC15 interface of the CS-MIPI-SC132 capture system module to the FPC28 interface of ADP-N1; 
3) SC132GS, a global shutter monochrome sensor from SmartSens’ SmartGS™ series. 

![CS-TX2-XAVIER-nCAM](./images/TX2_6CAM.png)

This is a cost-effective MIPI CSI interface camera module that supports up to 120FPS at 640×480 resolution or 30FPS at 1920×1080 resolution with global shutter recording. It has a built-in ISP for image processing, with adjustable brightness and contrast. The module uses uncompressed UYVY data format.


### System and Camera Driver Installation  

To make Jetson series boards compatible with the **CS-TX2-XAVIER-nCAM** module, you need a host machine running Ubuntu 18.04, which will be used to flash the Jetson TX2 system with **JetPack 4.6.1** (Ubuntu 18.04, Python 3.6). After flashing, replace and add the kernel package provided by the module manufacturer. The overall steps are as follows:  

1. **Download and Install SDK Manager on the Host Machine**  
   - SDK Manager is a tool released by NVIDIA for updating software and drivers for the Jetson platform series chips.  

2. **Connect Jetson TX2 to the Host Machine**  
   - Use a USB-to-MicroUSB cable to connect the Jetson TX2 to the host machine. Ensure that the Jetson TX2 is powered on.  

3. **Launch SDK Manager and Log into NVIDIA Account**  
   - Select the development board version and JetPack 4.6.1. Install all components, including DeepStream, CUDA-X, TensorRT, etc.  

4. **Follow the Instructions to Complete Flashing**  
   - Complete the flashing process as per the instructions provided by SDK Manager. Boot into the Ubuntu system and test whether all drivers and pre-installed software are working correctly.  

5. **Obtain and Install the Board Support Package (BSP) and Device Tree Blob (DTB)**  
   - Download the BSP and DTB from the GitHub repository provided by Zhong'an Yijia.  
   - **BSP (Board Support Package)**: A software suite that includes drivers, libraries, firmware, and configuration files specific to the hardware platform. BSP enables developers to access hardware resources such as GPIO, serial ports, and network interfaces.  
   - **DTB (Device Tree Blob)**: A binary file that describes the hardware platform. It contains information about the processor architecture, memory layout, peripherals, etc. The Linux kernel uses the DTB to dynamically configure hardware resources, allowing applications to access them.  

6. **Test Camera Connection**  
   - Use the command `dmesg | grep veye` to check if the SCCS132 camera model is detected.  

This process ensures that the Jetson TX2 is correctly configured to work with the **CS-TX2-XAVIER-nCAM** module, including its camera and hardware functionalities.

## Structure
### Overall structure


![Overall structure diagram](./images/Overall%20structure%20diagram.png)

### Structure of the mouse cage and excrement recovery box
The cage is made of five acrylic plates, forming a 2 × 2 structure. Each cage has four sides, no top and no bottom. The overall size is 24 cm × 24 cm × 35 cm, and the internal size of each cage is 10 cm × 10 cm × 20 cm. There is a mesh on the top of the excrement recovery box, and the mouse excrement can fall into the collection bag below through the mesh. The cage and excrement recovery box are detachable, which makes it easy to capture mice and clean up excrement after the experiment, while ensuring that the mice have enough space to move around.
![Separable design](./images/Separable%20design.png)

### Internal structure

![Internal structure](./images/Internal%20structure.gif)
