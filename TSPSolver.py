#!/usr/bin/python3

from copy import deepcopy
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy(self, time_allowance=60.0):
		pass
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound(self, time_allowance=60.0):
		bssf = self.defaultRandomTour(time_allowance)['soln']
		print("Random Algorithms found a BSSF of {}".format(bssf.cost))
		self.cities = self._scenario.getCities()
		self.numCities = len(self.cities)
		self.lowestBound = bssf.cost
		heap = []
		bestsFound = 0
		queueSize = 0
		totalStates = 0
		pruned = 0
		solutions = 0

		reducedMatrix, lowestBound = self.getBaseMatrix(self.cities)
		heapq.heappush(heap, tuple((lowestBound, reducedMatrix, self.cities[0], self.cities[1:], [self.cities[0]._index])))

		start = time.time()
		while len(heap) and time.time() - start < time_allowance:
			nextCity = heapq.heappop(heap)
			if nextCity[0] >= self.lowestBound:
				pruned += 1
				totalStates += 1
				continue
			else:
				for city in nextCity[3]:
					if self._scenario._edge_exists[nextCity[2]._index][city._index]:
						potential = self.getMatrix(city, nextCity)

						# if we have a completed tour
						if len(potential[3]) == 0:
							solutions += 1
							tour = potential[6]
							bssf = TSPSolution([self.cities[i] for i in tour])
							if bssf.cost < self.lowestBound:
								self.lowestBound = bssf.cost
								bestsFound += 1
								print("Found a BSSF of {}".format(bssf.cost))
						else:
							if potential[0] < self.lowestBound:
								heapq.heappush(heap, potential)
								queueSize = max(queueSize, len(heap))
								totalStates += 1
							else:
								pruned += 1
								totalStates += 1
			queueSize = max(queueSize, len(heap))
		
		end = time.time()
		results = {}
		results['cost'] = self.lowestBound
		results['time'] = end - start
		results['count'] = solutions
		results['soln'] = bssf
		results['max'] = queueSize
		results['total'] = totalStates
		results['pruned'] = pruned
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass

	def getBaseMatrix(self, cities):
		matrix = np.full((self.numCities, self.numCities), np.inf)

		for i in range(self.numCities):
			for j in range(self.numCities):
				if i != j:
					matrix[i][j] = cities[i].costTo(cities[j])

		lowestBound = 0

		for i in range(self.numCities):
			lowestBound += np.min(matrix[i])
			matrix[i] -= np.min(matrix[i])

		for i in range(self.numCities):
			lowestBound += np.min(matrix[:, i])
			matrix[:, i] -= np.min(matrix[:, i])

		return matrix, lowestBound

	def getMatrix(self, city, givenTuple):
		tupleCopy = deepcopy(givenTuple)
		matrix = tupleCopy[1]
		initialCost = matrix[tupleCopy[2]._index][city._index]
		reducedSum = 0

		matrix[tupleCopy[2]._index] = np.inf
		matrix[:, tupleCopy[2]._index] = np.inf
		matrix[city._index][tupleCopy[2]._index] = np.inf

		for row in range(matrix.shape[0]):
			rowMin = np.min(matrix[row])
			if rowMin == np.inf:
				continue
			matrix[row][city._index] -= rowMin
			reducedSum += rowMin

		for col in range(matrix.shape[1]):
			colMin = np.min(matrix[:, col])
			if colMin == np.inf:
				continue
			matrix[:, col] -= colMin
			reducedSum += colMin
		
		remainingCities = tupleCopy[3]
		remainingCities = [c for c in remainingCities if c._index != city._index]

		cost = tupleCopy[0] + initialCost + reducedSum

		return tuple((cost, matrix, city, remainingCities, tupleCopy[4] + [city._index]))