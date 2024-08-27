# Compiler
CXX = g++

# Compiler Flags
CXXFLAGS = -Wall -std=c++11

# Executable name
TARGET = main

# Source files
SRCS = ./helloworld.cc

# Object files
OBJS = $(SRCS:.cc=.o)

# Default target to compile app
all: $(TARGET)

# Linking object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule for compiling .cc files into .o files
$.o: $.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)