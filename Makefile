CXXFLAGS=-g -fopenmp
LDFLAGS=-fopenmp

all: train breed

breed: nnwork.o breed.o
	g++ -o breed nnwork.o breed.o -fopenmp

train: nnwork.o train.o
	g++ -o train nnwork.o train.o -fopenmp

clean:
	rm -f train breed *.o
