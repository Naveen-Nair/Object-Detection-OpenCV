1. install opencv and its dependencies

2. Use 'test/index.cpp' to check if the open
cd test
g++ -std=c++11 test.cpp -o test.out `pkg-config --cflags --libs opencv4`
./test.out
An image will pop up, or else there is some error

3. Check image processing
 cd imageProcessing
 g++ -std=c++17 -ggdb `pkg-config --cflags --libs opencv4` imageprocessing.cpp 1.preprocessing.cpp 2.edgeDetection.cpp 3.extractFeatures.cpp 4.buildKDtree.cpp 5.matchFeatures.cpp 6.boundingBoxes.cpp
 ./a.out
