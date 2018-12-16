# Rectangle Detector

This is a demo program for a method for realtime rectangle detection from an image. This program detects ALL rectangular shapes viewed from 3D perspective in real time, utilizing a GPU. The program is purely rule-based.

# Movies

See the following youtube videos to see some results.

* https://www.youtube.com/watch?v=HpaY0HOomAI
* https://www.youtube.com/watch?v=BLJEYui0XcY

# Running demo programs

You need to set up OpenCL runtime in order to run the programs. A CPU device should work, but it is pretty slow.

The following programs will be built.

### rect

This is a program for detecting rectangles in a still image.

```
Usage : ./rect <image file name> [device number] [output file name]
```

Available OpenCL devices and their numbers are displayed if you execute the program without any arguments.

### poly

This is a program for converting edges in the image to polyline. The result is written to output.png.

```
Usage : ./poly <image file name> [device number]
```


### vidrect

This is a program for detecting rectangles from a video.

```
Usage : ./vidrect [device number] [input video file] [output video file] [Horizontal AOV]
```

You can use camera as input. In that case, specify the camera by cam:<cam id>,<width>,<height> as an input.
Output can be displayed on an window by specifying - as an output.

The following command line captures the video from the first camera in 1280x720 resolution and output is shown in the window.
```
./vidrect 0 cam:0,1280,720 - 72
```

The following command line captures the video from input.mpg and output to output.mpg.
```
./vidrect 0 input.mpg output.mpg 72
```


### vidpoly

This is a program for converting edges in the video to polyline.


# Build

Please use cmake to build the program.

```sh
$ cd rectdetect-X.XX
$ mkdir build
$ cd build
$ cmake ..
$ make
```

```sh
D:\rectdetect-X.XX> mkdir build & cd build
D:\rectdetect-X.XX> cmake -G"Visual Studio 15 2017 Win64" ..
D:\rectdetect-X.XX> cmake --build . --config Release
```

In order to build the program with Microsoft Visual Studio, you need to edit the CMakeLists.txt.

# License

This software is distributed under the MIT license.

