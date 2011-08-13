CXXFLAGS=-g

all: train breed

breed: nnwork.o breed.o
	g++ -o breed nnwork.o breed.o

train: nnwork.o train.o
	g++ -o train nnwork.o train.o 

clean:
	rm -f train breed *.o
