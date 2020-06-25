# distutils: language=c++

class Event:

	#cdef int name, start, end
	
	def __init__(self, int n_name, int n_start, int n_end):
		self.name = n_name
		self.start = n_start
		self.end = n_end
	

	def char get_itr(self, Event other):
		a1 = start
		a2 = end;
		b1 = other.start;
		b2 = other.end;
		'''
		cdef int a1 = start;
		cdef int a2 = end;
		cdef int b1 = other.start;
		cdef int b2 = other.end;
		'''

		#before
		if (a2 < b1):
			return 0#'b';

		#meets
		if (a2 == b1):
			return 1;#'m';

		#overlaps
		if (a1 < b1 and a2 < b2 and b1 < a2):
			return 2;#'o';

		#during
		if (a1 < b1 and b2 < a2):
			return 3;#'d';

		#finishes
		if (b1 < a1 and a2 == b2):
			return 4;#'f';

		#starts
		if (a1 == b1 and a2 < b2):
			return 5;#'s';

		#equals
		return 6;#'e';
	
"""
def bool compareEvents(Event e1, Event e2):

    if(e1.start < e2.start):
		return true
    return e1.start == e2.start and e1.end < e2.end


#from libcpp.vector cimport vector

def read_sparse_matrix(filename, num_features):
#cdef vector[Event] read_sparse_matrix(str filename, int& num_features):

	event_list = []
	#cdef vector[Event] event_list;

	#open file
	#ifstream file (filename, ios::in | ios::binary);
	file = open(filename, "rb")

	byte = file.read(1)
	num_features = byte
	'''
	int num_f;
	#get number of features
	if (file.is_open())
	    file.read ((char*)&num_f, sizeof(num_f));
	num_features = num_f;
	'''

	#parse the rest of the file
	current_feature = -1;
	p1, p2 = 0,0
	#int current_feature = -1;
	#int p1, p2;

	byte = file.read(1)
	while (byte != ""):
	#while (file.peek()!=EOF):

		p1 = file.read(1)
		p2 = file.read(1)
	
	    #file.read ((char*)&p1, sizeof(p1));
	    #file.read ((char*)&p2, sizeof(p2));

	    if( p2 == 0 ):
	    	current_feature = p1;
	    else:
		    Event e = Event(current_feature, p1, p2);
		    #event_list.push_back(e);
		    event_list.append(e)
		
	
	return event_list

import numpy as np
#cimport numpy as np

def extract_itr_seq_into_counts(str txt_file):

	# get events from file
	#cdef int num_features;
	#cdef vector[Event] events = read_sparse_matrix(txt_file, num_features);
	events, num_features = read_sparse_matrix(txt_file)

	events.sort()
	#sort(events.begin(), events.end(), compareEvents);

	# get a list of all of the ITRs in the txt_file
	shape = (num_features, num_features, 7);
	itr_list = np.zeros(shape, np.int)
	#cdef np.ndarray itr_list = np.zeros(shape, np.int);

	for i in range(len(events)):
	#for i in range(events.size()):
		#cdef int j = i+1;
		j = i+1;
		while (j < len(events) and events[i].name != events[j].name):
		#while (j < events.size() and events[i].name != events[j].name):
			itr_name = events[i].get_itr(events[j]);
			#int itr_name = events[i].get_itr(events[j]);

			e1 = events[i].name;
			e2 = events[j].name;
			#cdef int e1 = events[i].name;
			#cdef int e2 = events[j].name;

			itr_list[e1][e2][itr_name] += 1;

			j += 1;
		
	
	return itr_list;
"""
