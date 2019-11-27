import os

from tcg.TemporalContextGraphIAD import TemporalContextGraphIAD

if __name__ == '__main__':
	input_dir = "/home/mbc2004/datasets/BlockMovingSep/txt_frames_1/0"

	print("dir exists: ", os.path.exists(input_dir))
	tcg_model = TemporalContextGraphIAD()

	tcg_model.learn_model(input_dir)
	#tcg_model.learn_model_from_files(input_dir)

	#tcg_model.output_graph("output")
	#tcg_model.print_edges()
	#tcg_model.print_nodes()