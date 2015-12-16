CXX = g++
CFLAGS = -std=c++11 -I/usr/local/include
LIBS = -L/usr/local/lib \
			 -lopencv_core \
			 -lopencv_highgui \
			 -lopencv_imgproc \
			 -lopencv_videoio \
			 -lopencv_imgcodecs \
			 -lopencv_calib3d \
			 -lopencv_features2d \
			 -lopencv_xfeatures2d \
			 -lopencv_flann \
			 -lopencv_line_descriptor \
			 -lopencv_stitching \
			 -larmadillo \
			 -lcvsba \
			 -lpcl_visualization
OBJS = stereo.o highgui.o kcluster.o

all: capture calibrate stereo AR ovr

capture: capture.cpp
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

calibrate: calibrate.cpp
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

stereo.o: stereo.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

highgui.o: highgui.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

kcluster.o: kcluster.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

stereo: $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

AR.o: AR.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

AR: AR.o highgui.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

imgproc.o: imgproc.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

ovr.o: ovr.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

test2.o: test2.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

ovr: ovr.o highgui.o imgproc.o test2.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o capture calibrate stereo AR ovr

run:
	make clean; make; ./stereo
