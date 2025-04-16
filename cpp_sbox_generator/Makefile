CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall
LDFLAGS = 

SRCS = main.cpp QuantumCircuit.cpp QuantumSimulator.cpp HybridReversibleSboxBuilder.cpp
OBJS = $(SRCS:.cpp=.o)
DEPS = Gate.h QuantumCircuit.h QuantumSimulator.h HybridReversibleSboxBuilder.h Constants.h

TARGET = sbox_gen

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean