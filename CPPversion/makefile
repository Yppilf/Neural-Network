# Compiler
CXX = g++

# Compiler Flags
CXXFLAGS = -Wall -Werror -std=c++11 -O2

# Directories
SRC_DIRS = . dataGetter
OBJ_DIR = obj

# Source files
SRCS = $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cc))

# Object files
OBJS = $(pathsubst %.cc, $(OBJ_DIR)/%.o, $(SRCS))

# Executable name
TARGET = main

# Default target to compile app
all: $(TARGET)

# Linking object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Rule for compiling .cc files into .o files
$.o: $.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)