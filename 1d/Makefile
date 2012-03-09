
all: main

main: main.o
	g++ -g -O3 -DNDEBUG -o $@ $<

main.o: main.cpp
	g++ -g -O3 -DNDEBUG -c -o $@ $<
