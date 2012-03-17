CFLAGS=-g
CXXFLAGS=-g

all: train breed

breed: nnwork.o breed.o
	g++ -o breed nnwork.o breed.o -fopenmp

train: nnwork.o train.o
	g++ -o train nnwork.o train.o -fopenmp

nnwork.o: nnwork.c

train.o: train.cpp

breed.o: breed.cpp

clean:
	rm -f train breed *.o
