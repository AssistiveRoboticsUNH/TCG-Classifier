
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <fstream>
#include <vector>


using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

class Event{
public:
	int name;
	int occurence;
	int start;
	int end;

	Event(int n_name, int n_start, int n_end){
		name = n_name;
		start = n_start;
		end = n_end;
	}

	char get_itr(Event other){
		int a1 = start;
		int a2 = end;
		int b1 = other.start;
		int b2 = other.end;

		//before
		if (a2 < b1)
			return 0;//'b';

		//meets
		if (a2 == b1)
			return 1;//'m';

		//overlaps
		if (a1 < b1 and a2 < b2 and b1 < a2)
			return 2;//'o';

		//during
		if (a1 < b1 and b2 < a2)
			return 3;//'d';

		//finishes
		if (b1 < a1 and a2 == b2)
			return 4;//'f';

		//starts
		if (a1 == b1 and a2 < b2)
			return 5;//'s';

		//equals
		return 6;//'e';
	}
};

bool compareEvents(Event e1, Event e2) 
{ 
    if(e1.start < e2.start)
		return true;
    return e1.start == e2.start and e1.end < e2.end;
} 



vector<Event> read_sparse_matrix(string filename, int& num_features){

	vector<Event> event_list;
	
	//open file
	ifstream file (filename, ios::in | ios::binary);

	//get number of features
	if (file.is_open())
	    file.read ((char*)&num_features, sizeof(num_features));

	//parse the rest of teh file
	int current_feature = -1;
	int p1, p2;

	while (file.peek()!=EOF)
	{
	    file.read ((char*)&p1, sizeof(p1));
	    file.read ((char*)&p2, sizeof(p2));

	    if( p2 == 0 ){
	    	current_feature = p1;
	    }else{
		    Event e = Event(current_feature, p1, p2);
		    event_list.push_back(e);
		}
	}
	return event_list;
}
/*
string extract_itr_seq(string txt_file, int& num_features){

	// get events from file
	vector<Event> events = read_sparse_matrix(txt_file, num_features);
	sort(events.begin(), events.end(), compareEvents);

	// get a list of all of the ITRs in the txt_file
	string itr_list = "";
	for (int i = 0; i < events.size(); i++){
		int j = i+1;
		while (j < events.size() and events[i].name != events[j].name){
			char itr_name = events[i].get_itr(events[j]);

			int e1 = events[i].name;
			int e2 = events[j].name;

			itr_list += to_string(e1)+itr_name+to_string(e2)+" ";
			j += 1;
		}
	}
	return itr_list;
}
*/

np::ndarray extract_itr_seq_into_counts(string txt_file){

	cout << "t0" << endl;

	// get events from file
	int num_features;
	vector<Event> events = read_sparse_matrix(txt_file, num_features);
	sort(events.begin(), events.end(), compareEvents);
	cout << "t1" << endl;

	// get a list of all of the ITRs in the txt_file
	//int array[num_features][num_features][7] itr_list = {};
	p::tuple shape = p::make_tuple(num_features, num_features, 7);
	np::dtype dt = np::dtype::get_builtin<int>();
	np::ndarray itr_list = np::zeros(shape, dt);
    //std::copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));

	cout << "num_features: " << num_features << endl;

	for (int i = 0; i < events.size(); i++){
		int j = i+1;
		while (j < events.size() and events[i].name != events[j].name){
			int itr_name = events[i].get_itr(events[j]);

			int e1 = events[i].name;
			int e2 = events[j].name;

			if(e1 > num_features or e2 > num_features)
				cout << e1 << " or " << e2<< " greater than " << num_features << endl;

			itr_list[e1][e2][itr_name] += 1;

			j += 1;
		}
	}
	return itr_list;
}
/*
int main(){
	read_sparse_matrix("test.b");
}
*/


BOOST_PYTHON_MODULE(itr_parser)
{
    using namespace boost::python;
    //def("extract_itr_seq", extract_itr_seq);
    def("extract_itr_seq_into_counts", extract_itr_seq_into_counts);
}
