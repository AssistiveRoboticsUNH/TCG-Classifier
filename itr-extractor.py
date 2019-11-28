from sets import Set

# I need to sort all of the indiividual actions by feature. Then
# I can get the ITR with the next feature for each each row. I 
# do not need to figure out the relationship between A0 and C7. Just the most adjacent values of A and C.
# progress from left to right means I can ignore reverses operations?

# need to figure this out ASAP!




class ITR_Extractor:

	class AtomicEvent():
		def __init__(self, name, occurence, start=-1, end=-1):
			self.name = name
			self.occurence = occurence
			self.start = start
			self.end = end

		def __lt__(self, other):
			if( self.start < other.start ):
				return True
			return self.start == other.start and self.end < other.end

		def get_itr(self, other):
			return self.get_itr_from_time(self.start, self.end, other.start, other.end)

		def get_itr_from_time(self, a1, a2, b1, b2):

			#before
			if (a2 < b1):
				return 'b'

			#meets
			if (a2 == b1):
				return 'm'

			#overlaps
			if (a1 < b1 and a2 < b2 and b1 < a2):
				return 'o'

			#during
			if (a1 < b1 and b2 < a2):
				return 'd'

			#finishes
			if (b1 < a1 and a2 == b2):
				return 'f'

			#starts
			if (a1 == b1 and a2 < b2):
				return 's'

			#equals
			if (a1 == b1 and a2 == b2):
				return 'eq'

			#startedBy
			if (a1 == b1 and b2 < a2):
				return 'si'

			#contains
			if (b1 < a1 and a2 < b2):
				return 'di'

			#finishedBy
			if (a1 < b1 and a2 == b2):
				return 'fi'

			#overlappedBy
			if (b1 < a1 and b2 < a2 and a1 < b2):
				return 'oi'

			#metBy
			if (b2 == a1):
				return 'mi'

			#after
			if (b2 < a1):
				return 'bi'





	def read_file(self, txt_file):
		
		events = {}
		for line in list(open(txt_file, 'r')):
			line = line.split()

			event_tokens = line[0].split('_')
			time = line[1]
			
			event_name = event_tokens[0]
			event_occur = int(event_tokens[1])
			event_bound = event_tokens[2]

			event_id = event_name+'_'+str(event_occur)
			if (event_id not in events):
				events[event_id] = self.AtomicEvent(event_name, event_occur)

			if(event_bound == 's'):
				events[event_id].start = time
			else:
				events[event_id].end = time

		return events.values()

	
			

	def learn_model_from_files(self, txt_file, label):

		# get events from file
		events = sorted(self.read_file(txt_file)) 

		# get a list of all of the ITRs in the txt_file
		itr_set = Set()

		for i in range(len(events)):

			j = i+1
			while(j < len(events) and events[j].name != events[i].name):
				itr_name = events[i].get_itr( events[j] )

				if('i' not in itr_name):
					e1 = events[i].name#+"_"+str(events[i].occurence) 
					e2 = events[j].name#+"_"+str(events[j].occurence)

					itr = (e1, itr_name, e2)
					itr_set.add(itr)

				j+=1

		# determine if those ITRS are already in TCG, if not add them, if they are increase their count
		for itr in itr_set:
			if(itr not in self.tcgs[label]):
				self.tcgs[label][itr] = 0
			self.tcgs[label] += 1
						
		# #maintain a record of what followed that ITR as n-grams

	def __init__(self, num_classes):
		self.num_classes = num_classes

		self.tcgs = []
		for i in num_classes:
			self.tcgs.append({})


if __name__ == '__main__':
	tcg = ITR_Extractor(2)

	tcg.learn_model_from_files("/home/mbc2004/datasets/BlockMovingSep/txt_frames_1/0/after/351_0.txt", 0)
	tcg.learn_model_from_files("/home/mbc2004/datasets/BlockMovingSep/txt_frames_1/0/after/352_0.txt", 0)
	tcg.learn_model_from_files("/home/mbc2004/datasets/BlockMovingSep/txt_frames_1/0/after/353_0.txt", 0)


		


