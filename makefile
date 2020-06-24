.DEFAULT_GOAL := all

# location of the Python header files
PYTHON_VERSION = 3.6
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)m

# location of the Boost Python include files and library
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib64
BOOST_INC2 = /usr/local/include/
BOOST_LIB2 = /usr/local/lib64

TARGET = itr_parser

CFLAGS = --std=c++11 -lboost_python -lboost_numpy -I$(BOOST_INC2) -L$(BOOST_LIB2)

$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -L$(BOOST_LIB2) -l:libboost_python-py$(subst .,,$(PYTHON_VERSION)).so -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so $(CFLAGS) 

$(TARGET).o: $(TARGET).cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp $(CFLAGS)

all: $(TARGET).so

clean: rm *.so *.o