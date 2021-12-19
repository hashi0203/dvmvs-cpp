CC := g++
CFLAGS := -Wall -std=c++11
OPENCV := `pkg-config --cflags --libs opencv4`
# OPENCV := -I/usr/local/include/opencv2 -I/usr/local/include/opencv -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui
EIGEN := -I /usr/include/eigen3

main: utils.o keyframe_buffer.o dataset_loader.o run-testing-online.o
	$(CC) $(CFLAGS) utils.o keyframe_buffer.o dataset_loader.o run-testing-online.o $(OPENCV)

run-testing-online.o: run-testing-online.cpp
	$(CC) $(CFLAGS) -c run-testing-online.cpp

utils.o: utils.cpp
	$(CC) $(CFLAGS) $(EIGEN) -c utils.cpp

keyframe_buffer.o: keyframe_buffer.cpp
	$(CC) $(CFLAGS) -c keyframe_buffer.cpp

dataset_loader.o: dataset_loader.cpp
	$(CC) $(CFLAGS) -c dataset_loader.cpp $(OPENCV)

test:
	$(CC) $(CFLAGS) $(EIGEN) test.cpp $(OPENCV)

conv-test:
	$(CC) $(CFLAGS) conv-test.cpp && ./a.out

torch-test:
	$(CC) $(CFLAGS) torch-test.cpp && ./a.out

mnasnet-test:
	$(CC) $(CFLAGS) mnasnet-test.cpp && ./a.out

model-test:
	$(CC) $(CFLAGS) model-test.cpp && ./a.out

clean:
	$(RM) *.o *.out