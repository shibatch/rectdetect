# Rectangle Detector

This is a demo program for a method for realtime rectangle detection from an image. This program detects ALL rectangular shapes viewed from 3D perspective in real time, utilizing a GPU. The program is purely rule-based.

# Movies

See the following youtube videos to see some results.

https://www.youtube.com/watch?v=HpaY0HOomAI
https://www.youtube.com/watch?v=BLJEYui0XcY

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
