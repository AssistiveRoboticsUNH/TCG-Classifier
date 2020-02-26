/*
#include <vector>
#include <iostream>
using namespace std;
*/
//#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <unordered_set>
#include <map>
#include <pthread.h>
//#include <vector>

//#include <boost/python/numpy.hpp>

//#include <boost/python/numeric.hpp>

//namespace np = boost::python::numpy;
/*
#define NUM_THREADS 5

struct thread_data{
	boost::python::list pairwise_projection;
	int projection_shape;
	boost::python::list itr_counts;
};


*/

s



void get_matches(boost::python::list const collapsed_projection, int projection_shape, boost::python::list itr_counts){

	for(int t = 0; t < projection_shape-1;  t++){
		std::map<int,int>::iterator loc = short_filters.find(boost::python::extract<int>(collapsed_projection[t]) );
		if(loc != short_filters.end() ){
			itr_counts[loc->second] += 1;
		}
		
		else {
			std::map<int,std::map<int,int> >::iterator loc_g = core_filters.find( boost::python::extract<int>(collapsed_projection[t]) );

			if (loc_g != core_filters.end() ){
				std::map<int,int> sub_map = loc_g->second;
				
				std::map<int,int>::iterator loc = sub_map.find( boost::python::extract<int>(collapsed_projection[t]) );
				if (loc != sub_map.end() ){
					itr_counts[loc->second] += 1;
				}
			}
		}
	}
}

/*
void get_all_matches(boost::python::list projection, int num_pw_comb, int proj_len, boost::python::list itr_counts){
	for (int pw_comb = 0; pw_comb < num_pw_comb; pw_comb++){
		auto pairwise_projection = projection[pw_comb];

		collapse(pairwise_projection, proj_len);
		
		//get the indexes of confounding features
		confound_ind = []
		for (int i = 0; i < confounding_filters; i++){
			conf_value = confounding_filters[i];
			pairwise_projection
			confound_ind.append( np.where(pairwise_projection == conf_value)[0] )
		}
		

		#remove the confounding variables
		collapsed_projection = np.delete(pairwise_projection, confound_ind)
		time_segments[0] += time.time() - t_i
		
	}
}
*/
/*
void collapse_and_count(boost::python::list pairwise_projection, int projection_shape, boost::python::list itr_counts){
			
	int new_shape = collapse(pairwise_projection, projection_shape);
	get_matches(pairwise_projection, new_shape, itr_counts);
}
*/
void *collapse_and_count_thread(void *thread_args){
	struct thread_data *my_data;
	my_data = (struct thread_data *) thread_args;
			
	int new_shape = collapse(my_data->pairwise_projection, my_data->projection_shape);
	get_matches(my_data->pairwise_projection, new_shape, my_data->itr_counts);
}

int collapse(boost::python::list pairwise_projection, int projection_shape){
	int new_shape = projection_shape;
	//get the indexes of confounding features
	for (int i = projection_shape-1; i >= 0; i--){
		if(confounding_filters.find( boost::python::extract<int>(pairwise_projection[i])) == confounding_filters.end() ){
			pairwise_projection.pop(i);
			new_shape--;
		}
	}
	return new_shape;
}

void thread(boost::python::list pairwise_projection, int projection_shape, boost::python::list itr_counts){
	td[0].pairwise_projection = pairwise_projection;
	td[0].projection_shape = projection_shape;
	td[0].itr_counts = itr_counts;

	pthread_create(&threads[0], NULL, collapse_and_count_thread, (void *)&td[0]);
}

/*
void join(){
	for(std::vector< boost::thread >::iterator t = threads.begin(); t != threads.end(); t++ ){
		t->join();
	}
}
*/
int main(){
	std::unordered_set<int> confounding_filters = std::unordered_set<int>({0, 3, 12, 15});
	/*
	int core_filters[13][2] =
		{
		{  9, -1 },//meets
		{  6, -1 },//metBy
		{  5, 11 },//starts
		{  5, 14 },//startedBy
		{  7, 10 },//finishes
		{ 13, 10 },//finishedBy
		{ 13, 11 },//overlaps
		{  7, 14 },//overlapedBy
		{  7, 11 },//during
		{ 13, 14 },//contains
		{  8,  1 },//before
		{  2,  4 },//after
		{  5, 10 },//equals
		};
	*/
	std::map<int, int> short_filters;
	std::map<int, std::map<int, int> > core_filters;
	short_filters = std::map<int, int>();
	short_filters[9] = 0;
	short_filters[6] = 1;

	core_filters = std::map<int, std::map<int, int> >();
	core_filters[8] = std::map<int, int>();
	core_filters[8][1] = 10;

	core_filters[2] = std::map<int, int>();
	core_filters[2][4] = 11;

	core_filters[5] = std::map<int, int>();
	core_filters[5][10] = 12;
	core_filters[5][11] = 2;
	core_filters[5][14] = 3;

	core_filters[7] = std::map<int, int>();
	core_filters[7][10] = 4;
	core_filters[7][11] = 8;
	core_filters[7][14] = 7;

	core_filters[13] = std::map<int, int>();
	core_filters[13][10] = 5;
	core_filters[13][11] = 9;
	core_filters[13][14] = 6;
}


using namespace boost::python;

BOOST_PYTHON_MODULE(itr_matcher)
{
    
    class_<ITRMatcher>("ITRMatcher", init<int>())
        .def("get_matches", &ITRMatcher::get_matches)
        .def("collapse", &ITRMatcher::collapse)
        .def("collapse_and_count", &ITRMatcher::collapse_and_count)

    ;
}
