

# I need to sort all of the indiividual actions by feature. Then
# I can get the ITR with the next feature for each each row. I 
# do not need to figure out the relationship between A0 and C7. Just the most adjacent values of A and C.
# progress from left to right means I can ignore reverses operations?

# need to figure this out ASAP!







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


class ITR_Extractor:

	def read_file(txt_file):
		
		events = {}
		for line in list(open(txt_file, 'r')):
			line = line.split()

			event_tokens = line[0].split('_')
			time = line[1]
			
			event_name = event_tokens[0]
			event_occur = int(event_tokens[1])
			event_bound = event_tokens[2]

			event_id = event_name+'_'+str(event_occur)
			if (event_id not in events)
				events[event_id] = AtomicEvent(event_name, event_occur)

			if(event_bound == 's'):
				events[event_id].start = time
			else:
				events[event_id].end = time

		return events.values()
			

	def learn_model_from_files(txt_file, label):

		# get events from file
		events = read_file(txt_file)

		# get a list of all of the ITRs in the txt_file
		for e in sorted(events):
			itr = self.determine_itr(e, events)

		# determine if those ITRS are already in TCG, if not add them, if they are increase their count



		# #maintain a record of what followed that ITR as n-grams





		


