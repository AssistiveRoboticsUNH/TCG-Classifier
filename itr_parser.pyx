# distutils: language=c++


#include <boost/python.hpp>
import numpy as np

cdef class Event:

	cdef int name, start, end
	
	def __init__(self, int n_name, int n_start, int n_end):
		self.name = n_name
		self.start = n_start
		self.end = n_end
	

	def char get_itr(self, Event other):
		cdef int a1 = start;
		cdef int a2 = end;
		cdef int b1 = other.start;
		cdef int b2 = other.end;

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
	

def bool compareEvents(Event e1, Event e2):

    if(e1.start < e2.start):
		return true
    return e1.start == e2.start and e1.end < e2.end


from libcpp.vector cimport vector

cdef vector[Event] read_sparse_matrix(string filename, int& num_features):

	cdef vector[Event] event_list;
	
	#open file
	ifstream file (filename, ios::in | ios::binary);

	int num_f;
	#get number of features
	if (file.is_open())
	    file.read ((char*)&num_f, sizeof(num_f));
	num_features = num_f;

	#parse the rest of the file
	int current_feature = -1;
	int p1, p2;

	while (file.peek()!=EOF):
	
	    file.read ((char*)&p1, sizeof(p1));
	    file.read ((char*)&p2, sizeof(p2));

	    if( p2 == 0 ):
	    	current_feature = p1;
	    else:
		    Event e = Event(current_feature, p1, p2);
		    event_list.push_back(e);
		
	
	return event_list



np::ndarray extract_itr_seq_into_counts(string txt_file):

	# get events from file
	cdef int num_features;
	cdef vector<Event> events = read_sparse_matrix(txt_file, num_features);
	sort(events.begin(), events.end(), compareEvents);

	# get a list of all of the ITRs in the txt_file
	p::tuple shape = p::make_tuple(num_features, num_features, 7);
	np::dtype dt = np::dtype::get_builtin<int>();
	np::ndarray itr_list = np::zeros(shape, dt);

	for i in range(events.size()):
		cdef int j = i+1;
		while (j < events.size() and events[i].name != events[j].name):
			int itr_name = events[i].get_itr(events[j]);

			cdef int e1 = events[i].name;
			cdef int e2 = events[j].name;

			itr_list[e1][e2][itr_name] += 1;

			j += 1;
		
	
	return itr_list;

