CFLAGS=-g
CXXFLAGS=-g

all: train breed

breed: nnwork.o breed.o
	g++ -o breed nnwork.o breed.o

train: nnwork.o train.o
	g++ -o train nnwork.o train.o 

nnwork.o: nnwork.c
	cc -c nnwork.c

train.o: train.cpp
	g++ -c train.cpp

breed.o: breed.cpp
	g++ -c breed.cpp

clean:
	rm -f train breed *.o
